from collections import defaultdict
from typing import Any, List, Tuple, Dict
import os

from Core.Index.Tree import NodeType
from Core.rag.base_rag import BaseRAG
from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.configs.rag.graph_config import GraphRAGConfig
from Core.Index.GBCIndex import GBC
from Core.prompts.gbc_prompt import (
    QuestionEntity,
    QuestionEntityExtraction,
    QUESTION_EE_PROMPT,
    QUESTION_ENTITY_TYPES,
)
from Core.Index.Graph import Entity
from Core.utils.table_utils import table2text
from Core.utils.utils import TextProcessor

from Core.rag.gbc_utils import enhance_graph_with_semantic_links


import json
import networkx as nx

import logging

log = logging.getLogger(__name__)


class GraphRAG(BaseRAG):
    """
    Graph RAG (Graph-Based Contextual Retrieval Augmented Generation) class.
    This class is designed to handle the retrieval and generation of responses
    based on a graph-based context.
    """

    def __init__(
        self,
        llm: LLM,
        vlm: VLM,
        config: GraphRAGConfig,
        gbc_index: GBC,
    ):
        super().__init__(
            llm,
            name="Graph RAG",
            description="Graph-Based Contextual Retrieval Augmented Generation",
        )
        self.vlm = vlm
        self.cfg = config
        if not gbc_index:
            raise ValueError("GBC index must be provided for GBCRAG.")
        self.gbc_index = gbc_index
        self.embedder = self.gbc_index.embedder if self.gbc_index else None

        # Graph RAG config
        self.threshold_e = self.cfg.sim_threshold_e
        self.max_retry = self.cfg.max_retry
        self.topk_docs = self.cfg.topk_docs

        # Graph Augmentation parameters
        self.x_percentile = self.cfg.x_percentile
        self.alpha = self.cfg.alpha
        self.topk_ent = self.cfg.topk_ent

        # Enhance graph for PageRank
        self.graph = self.gbc_index.GraphIndex.kg
        self.enhanced_graph = enhance_graph_with_semantic_links(
            graph=self.graph, embedder=self.embedder, x_percentile=self.x_percentile
        )

    def _extract_entities(self, query: str) -> List[Any]:
        """
        Extracts entities from the query string.
        This method should be implemented to extract relevant entities
        that can be used for retrieval.
        """
        # Placeholder for entity extraction logic
        prompt = QUESTION_EE_PROMPT.format(
            input_text=query, entity_types=", ".join(QUESTION_ENTITY_TYPES)
        )
        try:
            res: QuestionEntityExtraction = self.llm.get_json_completion(
                prompt, QuestionEntityExtraction
            )
            if res and res.entities:
                entities = res.entities
                entities_name = [entity.entity_name for entity in entities]
                log.info(f"Extracted entities: {entities_name}")
                return entities
            else:
                log.info("No entities extracted from the query.")

        except Exception as e:
            log.error(f"Error during entity extraction: {e}")

        log.info("Use the question as the entity.")
        res_entities = [Entity(entity_name=query, entity_type="Question")]
        return res_entities

    def _get_entity_embed_text(self, entity: QuestionEntity) -> str:
        return f"Name: {entity.entity_name}\nType: {entity.entity_type}"

    def _entity_map(self, entities: List[str]) -> Dict[str, List[str]]:
        """
        Maps entities to their corresponding IDs in the GBC index.
        Use vdb to find the entity in GBC index.
        """
        entities_str = [self._get_entity_embed_text(entity) for entity in entities]
        Qent_GBCent_map = defaultdict(list)
        for ent_str in entities_str:
            query_res = self.gbc_index.entity_vdb.search(query_text=ent_str, top_k=1)
            retrieve_name = query_res[0]["metadata"].get("entity_name")
            retrieve_type = query_res[0]["metadata"].get("entity_type")
            node_name = self.gbc_index.GraphIndex.get_node_name_from_str(
                retrieve_name, retrieve_type
            )
            Qent_GBCent_map[ent_str].append(node_name)
            log.info(f"Entity '{ent_str}' mapped to GBC entity: {node_name}")

        return Qent_GBCent_map

    def graph_reranker(
        self,
        ent_map: Dict[str, List[str]],
    ) -> List[Tuple[int, float]]:
        ents_sim = {}
        for q_ent_name, gbc_ents in ent_map.items():
            for node_name in gbc_ents:
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

        pagerank_scores = nx.pagerank(
            self.enhanced_graph,
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
        # 3.2 初始化一个字典来累加每个 tree_id 的分数
        tree_node_scores = defaultdict(float)

        # 3.3 遍历 PageRank 的结果
        for entity_name, score in pagerank_scores.items():
            # 从图中获取该实体节点的属性
            # .get('source_ids', set()) 是一种安全的方式，防止节点没有该属性时出错
            node_attributes = self.graph.nodes.get(entity_name, {})
            source_ids = node_attributes.get("source_ids", set())

            for source_id in source_ids:
                tree_node_scores[source_id] += score

        # 4.1 将聚合后的分数字典转换为 (id, score) 的元组列表
        aggregated_ranked_list = list(tree_node_scores.items())

        # 4.2 按聚合后的分数降序排序
        sorted_ranked_list = sorted(
            aggregated_ranked_list, key=lambda item: item[1], reverse=True
        )

        sorted_ranked_list = sorted_ranked_list[: self.topk_docs]

        return sorted_ranked_list, res_entities

    def get_graph_info(
        self,
        ent_map: Dict[str, List[str]],
    ) -> None:
        # 2.1 PPR to rank the most relevant TreeNodes in the subtree.
        graph_rerank_res, res_entities = self.graph_reranker(ent_map)

        tree_node_ids = [node_id for node_id, _ in graph_rerank_res]

        # 3. The final graph data info contain TreeNodes and Subgraph info (use Entities node name instead)
        graph_info = {"TreeNode_ids": tree_node_ids, "EntNode_name": res_entities}
        return graph_info

    def _retrieve(
        self,
        query: str,
    ) -> None:
        """
        GBC retrieval following the steps:
        1. Extract entities from the query.
        2. Get the section nodes based on the entities.
        3. Use LLM to select the most relevant section based on the query and Section info.
        4. Use graph-based retrieval on the subgraph projected by the subtree (Select Section).

        iter_context: IterationStep, Iteration context for the current step.
        """
        # 1. Extract entities from the query
        # For the first iteration, extract entities from the query
        query_entities = self._extract_entities(query)
        if not query_entities:
            return {"TreeNode_ids": [], "EntNode_name": []}

        # 2. Get the entity mapping to GBC entities
        Qent_GBCent_map = self._entity_map(query_entities)

        # 4. Graph-based retrieval on subgraph projected by the subtree (Select Section)
        graph_info = self.get_graph_info(Qent_GBCent_map)
        return graph_info

    def _create_augmented_prompt(self, query: str, graph_info: dict):
        # 1. get retrieval data

        TreeNode_ids = graph_info.get("TreeNode_ids")
        # Get the data for the selected TreeNodes and Entity nodes
        Tree_data = self.gbc_index.TreeIndex.get_nodes_data(TreeNode_ids)

        # context_text = "Please refer to the following background information to answer the question. And you should try you best to given the answer\n\n"
        context_text = "Please refer to the following background information to answer the question. You should answer the question based on the provided information. Don't make up any information.\n\n--- Background Information ---\n"
        context_images = []
        question_text = f"--- User Question ---\n{query}\n\n"

        if Tree_data is None or len(Tree_data) == 0:
            context_text += "No relevant documents found.\n"
            context_text += question_text
            return context_text, context_images

        for node_data in Tree_data:
            node_type = node_data.get("type", "text")
            if node_type == NodeType.IMAGE:
                image_path = node_data.get("img_path", "")
                if image_path and os.path.exists(image_path):
                    context_images.append(image_path)
                context_text += (
                    f"Image: A relevant image is provided at the path: {image_path}\n"
                )
            elif node_type == NodeType.TABLE:
                node_data["content"] = table2text(node_data)
                context_text += f"Table: {node_data['content']}\n"
            else:
                context_text += f"Text: {node_data['content']}\n"

        # limit to 2 images as the VLM only support 2 images at once
        context = TextProcessor.split_text_into_chunks(
            context_text, max_length=self.llm.config.max_tokens - 400
        )
        context_text = context[0]

        context_images = context_images[:2]
        context_text += question_text
        return context_text, context_images

    def generation(self, query: str, query_output_dir: str):
        # Initialize the first iteration step
        cnt = 0
        while cnt < self.max_retry:
            cnt += 1
            log.info(f"Iteration {cnt} for query: {query}")

            # GBC retrieval process
            graph_info = self._retrieve(query)

            context_text, context_images = self._create_augmented_prompt(
                query, graph_info
            )
            if len(context_images) > 0:
                # if there are images, use VLM to generate the answer
                final_answer = self.vlm.generate(
                    prompt_or_memory=context_text, images=context_images
                )
            else:
                # if no images, fallback to LLM generation
                final_answer = self.llm.get_completion(
                    context_text, json_response=False
                )

            retrieval_ids = self._save_retrieval_res(
                graph_info, query_output_dir=query_output_dir
            )
            return final_answer, retrieval_ids

    def _save_retrieval_res(self, graph_info, query_output_dir: str):
        retrieval_ids = []

        # direct save the context to a json file
        retrieval_save_res = query_output_dir / "retrieval_res.json"
        with open(retrieval_save_res, "w", encoding="utf-8") as f:
            json.dump(graph_info, f, indent=2, ensure_ascii=False)
        log.info(f"Retrieval results saved to {retrieval_save_res}")

        # use the graph nodes as retrieval ids
        retrieval_ids = graph_info.get("TreeNode_ids", [])
        return retrieval_ids

    def close(self):
        self.embedder.close()
        return super().close()
