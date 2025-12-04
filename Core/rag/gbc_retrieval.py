from typing import List, Tuple, Dict
import networkx as nx
import logging

from Core.provider.embedding import TextEmbeddingProvider, MMRerankerProvider
from Core.provider.rerank import TextRerankerProvider
from Core.Index.Tree import TreeNode, NodeType
from Core.utils.table_utils import table2text
from Core.utils.utils import num_tokens, TextProcessor
from Core.prompts.gbc_prompt import TEXT_RERANKER_PROMPT, MM_RERANKER_INSTRUCTION
from Core.rag.gbc_utils import (
    enhance_graph_with_semantic_links,
    merge_ranker_scores,
    calculate_skyline,
)

log = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        varient: str,
        reranker: TextRerankerProvider,
        # mm_reranker: MMRerankerProvider,
        embedder: TextEmbeddingProvider,
        alpha: float = 0.85,
        topk_ent: int = 20,
        x_percentile: int = 90,
        topk: int = 10,
    ):
        self.varient = varient
        self.reranker: TextRerankerProvider = reranker
        # self.mm_reranker: MMRerankerProvider = mm_reranker
        self.embedder: TextEmbeddingProvider = embedder
        self.alpha: float = alpha
        self.topk_ent: int = topk_ent
        self.x_percentile: int = x_percentile
        self.topk: int = topk

    def text_reranker(
        self, subtree_nodes: List[TreeNode], query: str
    ) -> List[Tuple[int, float]]:
        """
        Rerank the original text in the subtree based on their relevance to the query.
        Use Reranker model
        """
        doc_text = []
        tree_ids = [node.index_id for node in subtree_nodes]
        for node in subtree_nodes:
            node_type = node.type
            if node_type == NodeType.TABLE:
                # Convert table to text
                text = table2text(node.meta_info.__dict__)
            else:
                text = node.meta_info.content
            if num_tokens(text) > self.reranker.max_length - 1000:
                text = TextProcessor.split_text_into_chunks(
                    text=text, max_length=self.reranker.max_length - 1000
                )
                text = text[0]  # use the first chunk
            doc_text.append(text)

        scores = self.reranker.rerank(
            query=query, documents=doc_text, instruction=TEXT_RERANKER_PROMPT
        )
        self.reranker.clean_cache()
        ranked_res = sorted(zip(tree_ids, scores), key=lambda x: x[1], reverse=True)

        return ranked_res

    # def multimodal_reranker(
    #     self, subtree_nodes: List[TreeNode], query: str
    # ) -> List[Tuple[int, float]]:
    #     doc_mm = []
    #     tree_ids = [node.index_id for node in subtree_nodes]
    #     for node in subtree_nodes:
    #         node_type = node.type
    #         tmp_dict = {}
    #         if node_type == NodeType.TABLE or node_type == NodeType.IMAGE:
    #             img_path = node.meta_info.img_path
    #             tmp_dict["img_path"] = img_path
    #         text = node.meta_info.content
    #         tmp_dict["content"] = text
    #         tmp_dict["node_id"] = node.index_id
    #         doc_mm.append(tmp_dict)

    #     scores = self.mm_reranker.rerank_documents(
    #         query=query, doc_list=doc_mm, instruction=MM_RERANKER_INSTRUCTION
    #     )
    #     ranked_res = sorted(zip(tree_ids, scores), key=lambda x: x[1], reverse=True)
    #     # Return the ranked results as a list of tuples (tree_id, score)
    #     return ranked_res

    def graph_reranker(
        self,
        subgraph: nx.Graph,
        ent_map: Dict[str, List[str]],
        subtree_nodes: List[TreeNode],
    ) -> Tuple[List[Tuple[int, float]], List[str]]:
        ents_sim = {}
        for q_ent_name, gbc_ents in ent_map.items():
            for node_name in gbc_ents:
                if node_name not in subgraph.nodes():
                    continue
                # compute similarity between the query entity and the GBC entity
                sim = self.embedder.compute_texts_sim(q_ent_name, node_name)
                positive_sim = (sim + 1) / 2
                ents_sim[node_name] = ents_sim.get(node_name, 0.0) + positive_sim

        total_sim_sum = sum(ents_sim.values())

        # Normalize the similarity scores
        personalization_vector = {}
        if total_sim_sum > 0:
            for ent, score in ents_sim.items():
                personalization_vector[ent] = score / total_sim_sum
        else:
            return [], []

        pagerank_alpha: float = self.alpha
        # --- 步骤 1 & 2: 增强图并计算 PageRank 分数 ---
        enhanced_graph = enhance_graph_with_semantic_links(
            graph=subgraph, embedder=self.embedder, x_percentile=self.x_percentile
        )

        pagerank_scores = nx.pagerank(
            enhanced_graph,
            alpha=pagerank_alpha,
            personalization=personalization_vector,
            weight="weight",  # 可以考虑使用相似度作为边的权重
        )

        # sort the ranked list by score and filter by topk_ent
        res_entities = [(ent, score) for ent, score in pagerank_scores.items()]
        res_entities = sorted(res_entities, key=lambda x: x[1], reverse=True)
        res_entities = res_entities[: self.topk_ent]
        res_entities = [ent for ent, _ in res_entities]
        log.info(
            f"Graph reranker: Retrieved {len(res_entities)} entities with topk={self.topk_ent}."
        )

        # --- 步骤 3 & 4: 聚合分数到 Tree Node ID 并排序返回 ---
        # 3.1 创建一个目标 tree_id 的集合，用于快速查找
        target_tree_ids = {node.index_id for node in subtree_nodes}
        # 3.2 初始化一个字典来累加每个 tree_id 的分数
        tree_node_scores = {}
        for tree_id in target_tree_ids:
            tree_node_scores[tree_id] = 0.0

        # 3.3 遍历 PageRank 的结果
        for entity_name, score in pagerank_scores.items():
            # 从图中获取该实体节点的属性
            # .get('source_ids', set()) 是一种安全的方式，防止节点没有该属性时出错
            node_attributes = subgraph.nodes.get(entity_name, {})
            source_ids = node_attributes.get("source_ids", set())

            # 遍历该实体的所有 source_id
            for source_id in source_ids:
                # 如果这个 source_id 是我们关心的目标 tree_id
                if source_id in target_tree_ids:
                    # 将该实体的 PageRank 分数累加到对应的 tree_node_scores 中
                    tree_node_scores[source_id] += score

        # 4.1 将聚合后的分数字典转换为 (id, score) 的元组列表
        aggregated_ranked_list = list(tree_node_scores.items())

        # 4.2 按聚合后的分数降序排序
        sorted_ranked_list = sorted(
            aggregated_ranked_list, key=lambda item: item[1], reverse=True
        )

        return sorted_ranked_list, res_entities

    def skyline_filter(
        self,
        sub_query: str,
        subtree_nodes: List[TreeNode],
        subgraph: nx.Graph,
        start_ent_map: Dict[str, List[str]],
    ) -> Tuple[List[int], List[str]]:
        if len(subtree_nodes) == 0:
            log.info("No subtree nodes available for reranking.")
            return [], []

        if self.varient == "wo_graph":
            log.info("Variant 'wo_graph' selected: Skipping graph reranker.")
            # Only use text reranker
            text_rerank_res = self.text_reranker(subtree_nodes, sub_query)
            tree_node_ids = [node_id for node_id, _ in text_rerank_res[: self.topk]]

            # use start_ent_map as the returned entities
            res_entities = set()
            for _, gbc_ents in start_ent_map.items():
                for node_name in gbc_ents:
                    res_entities.add(node_name)
            res_entities = list(res_entities)

            return tree_node_ids, res_entities

        if self.varient == "wo_text":
            log.info("Variant 'wo_text' selected: Skipping text reranker.")
            # Only use graph reranker
            graph_rerank_res, res_entities = self.graph_reranker(
                subgraph, start_ent_map, subtree_nodes
            )
            tree_node_ids = [node_id for node_id, _ in graph_rerank_res[: self.topk]]

            return tree_node_ids, res_entities

        # 2. Use Three Layer Reranker to select most relevant TreeNodes in the subtree.
        # 2.1 PPR to rank the most relevant TreeNodes in the subtree.
        graph_rerank_res, res_entities = self.graph_reranker(
            subgraph, start_ent_map, subtree_nodes
        )
        
        # tmp_test
        if len(graph_rerank_res) < self.topk:
            log.warning(
                f"Graph reranker returned only {len(graph_rerank_res)} nodes, "
                f"which is less than topk={self.topk}."
            )
            tree_node_ids = [node_id for node_id, _ in graph_rerank_res]
            return tree_node_ids, res_entities
        

        # 2.2 Rerank with text reranker model.
        text_rerank_res = self.text_reranker(subtree_nodes, sub_query)

        # 2.3 Rerank with Multimodal method.
        # mm_rerank_res = self.multimodal_reranker(subtree_nodes, sub_query)

        # Combine the results from three rerankers
        # merged_scores: Dict[int, List[float]] = merge_ranker_scores(
        #     graph_rerank_res, text_rerank_res, mm_rerank_res
        # )

        merged_scores: Dict[int, List[float]] = merge_ranker_scores(
            graph_rerank_res, text_rerank_res
        )
        sel_tree_nodes = calculate_skyline(merged_scores)
        tree_node_ids = [node["node_id"] for node in sel_tree_nodes]

        if (
            len(tree_node_ids) < self.topk
            and len(merged_scores) >= self.topk
        ):
            log.info(
                f"Skyline returned only {len(tree_node_ids)} nodes. "
                f"Activating fallback to meet minimum of {self.topk}."
            )

            # --- MODIFICATION START ---

            # 1. Get IDs from the initial skyline result
            final_node_ids = set(tree_node_ids)

            # 2. Get top 5 from graph reranker
            # Assuming the score is the first value in the list, sort descending
            top_5_graph = graph_rerank_res[:5]
            for node_id, _ in top_5_graph:
                final_node_ids.add(node_id)

            # 3. Get top 5 from text reranker
            top_5_text = text_rerank_res[:5]
            for node_id, _ in top_5_text:
                final_node_ids.add(node_id)

            # 4. Rebuild the sel_tree_nodes list from the unique IDs
            # Create a quick lookup map for subtree_nodes by their ID
            tree_node_ids = list(final_node_ids)

            log.info(f"Fallback resulted in {len(tree_node_ids)} unique nodes.")
            # --- MODIFICATION END ---

        return tree_node_ids, res_entities
