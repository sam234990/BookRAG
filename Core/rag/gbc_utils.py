from typing import List, Optional, Dict, Any, Tuple, Union
from pydantic import BaseModel, Field
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
from Core.Index.Tree import TreeNode, DocumentTree, NodeType
from Core.provider.embedding import TextEmbeddingProvider
from Core.rag.gbc_plan import Filter, PlanResult
import math
import logging


log = logging.getLogger(__name__)


class SubStep(BaseModel):
    sub_query: str
    sub_number: int
    gbc_entity_map: Dict[str, List[str]] = Field(default_factory=dict)
    linked_tree_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    linked_section_ids: List[int] = Field(default_factory=list)

    supplementary_ids: List[int] = Field(default_factory=list)
    selected_explanation: str = ""

    # store the retrieval info for visualization
    retrieval_sec_ids: List[int] = Field(default_factory=list)
    retrieval_nodes: Union[str | List[Any]] = Field(default_factory=list)
    iteration_text_nodes: Union[str | List[Any]] = Field(default_factory=list)
    iteration_image_nodes: Union[str | List[Any]] = Field(default_factory=list)
    iteration_graph_nodes: Union[str | List[Any]] = Field(default_factory=list)

    partial_answers: Union[str | List[Any]] = Field(default_factory=list)
    generated_answer: str = ""


class GBCRAGContext(BaseModel):
    query: str
    used_sec_ids: List[int] = Field(default_factory=list)
    all_sec_ids: List[int] = Field(default_factory=list)

    # Plan
    plan: PlanResult = None

    # Interation/ subquestion result
    iterations: List[SubStep] = Field(default_factory=list)

    # Output
    final_answer: str = ""


# --- Pydantic Models for Structured Output (已更新) ---


def filter_tree_nodes(
    tree_index: DocumentTree,
    filters: List[Filter],
) -> List[TreeNode]:
    filtered_nodes = tree_index.get_nodes(hasRoot=False)
    for f in filters:
        f_type = f.filter_type
        f_value = f.filter_value
        if f_type == "image" or f_type == "table":
            f_type = NodeType(f_type)
            filtered_nodes = [node for node in filtered_nodes if node.type == f_type]
        elif f_type == "page":
            if f_value is None:
                continue
            if "-" in f_value:
                try:
                    start_page, end_page = map(int, f_value.split("-"))
                    filtered_nodes = [
                        node
                        for node in filtered_nodes
                        if start_page <= node.meta_info.page_idx + 1 <= end_page
                    ]
                except ValueError:
                    log.warning(f"Invalid page range format: {f_value}")
                    continue
            else:
                try:
                    target_page = int(f_value)
                    filtered_nodes = [
                        node
                        for node in filtered_nodes
                        if node.meta_info.page_idx + 1 == target_page
                    ]
                except ValueError:
                    log.warning(f"Invalid page number format: {f_value}")
                    continue
        elif f_type == "section":
            matched_nodes = []
            if f_value is None:
                continue
            for node in filtered_nodes:
                node_idx = node.index_id
                root_path = tree_index.get_path_from_root(node_idx)
                title_str = ""
                for pnode in root_path:
                    if pnode.type == NodeType.TITLE:
                        title_str += pnode.meta_info.content.lower()
                if f_value.lower() in title_str:
                    matched_nodes.append(node)
            filtered_nodes = matched_nodes
        else:
            log.warning(f"Unknown filter type: {f_type}")
            continue

    return filtered_nodes


def merge_ranker_scores(
    *ranker_scores_lists: List[Tuple[int, float]]
) -> Dict[int, List[float]]:
    """
    将多个评分器（ranker）的评分列表合并为一个字典。

    Args:
        *ranker_scores_lists: 任意数量的评分列表。
            每个列表的格式为 [(node_id, score), ...]。

    Returns:
        一个字典，将每个 node_id 映射到一个包含所有评分器分数的列表。
        格式: {node_id: [score1, score2, ...]}
    """
    num_rankers = len(ranker_scores_lists)
    # 使用 lambda 创建一个长度为 num_rankers 的默认值列表
    merged_scores = defaultdict(lambda: [0.0] * num_rankers)

    # 遍历每个评分列表并填充分数
    for i, scores_list in enumerate(ranker_scores_lists):
        for node_id, score in scores_list:
            merged_scores[node_id][i] = score

    return dict(merged_scores)


def calculate_skyline(
    merged_scores: Dict[int, List[float]],
) -> List[Dict[str, Union[int, List[float]]]]:
    """
    根据合并后的分数计算 skyline 集合。
    一个点 p 被另一个点 s "支配" (dominate)，当且仅当 s 在所有维度上都不劣于 p，
    并且至少在一个维度上严格优于 p。

    Args:
        merged_scores: 一个字典，包含节点ID及其多维度的分数。
            格式: {node_id: [score1, score2, ...]}

    Returns:
        skyline_nodes: 不被任何其他节点支配的节点列表。
            格式: [{'node_id': id, 'dims': [score1, score2, ...]}, ...]
    """
    # 将字典转换为元组列表 (node_id, scores)
    nodes = list(merged_scores.items())

    # 如果节点数少于2，则它们都在skyline中
    if len(nodes) < 2:
        return [{"node_id": nid, "dims": scores} for nid, scores in nodes]

    # 获取维度数量
    num_dims = len(nodes[0][1])

    # 1. 根据第一个维度的分数进行降序排序
    # 这是一种常见的优化策略，可以提高剪枝效率
    nodes.sort(key=lambda item: item[1][0], reverse=True)

    skyline = []
    for p_id, p_scores in nodes:
        is_dominated = False
        # 2. 检查当前节点 p 是否被已有的 skyline 节点 s 支配
        for s_node in skyline:
            s_scores = s_node["dims"]

            # s 在所有维度上都不劣于 p
            all_dims_ge = all(s_scores[i] >= p_scores[i] for i in range(num_dims))
            # s 至少在一个维度上严格优于 p
            any_dim_gt = any(s_scores[i] > p_scores[i] for i in range(num_dims))

            if all_dims_ge and any_dim_gt:
                is_dominated = True
                break  # 已被支配，无需再比较

        # 3. 如果未被任何 skyline 中的节点支配，则将其加入 skyline
        if not is_dominated:
            new_skyline = []
            for s_node in skyline:
                s_scores = s_node["dims"]
                # 检查 s 是否被 p 支配
                if not (
                    all(p_scores[i] >= s_scores[i] for i in range(num_dims))
                    and any(p_scores[i] > s_scores[i] for i in range(num_dims))
                ):
                    new_skyline.append(s_node)

            # 将 p 加入新的 skyline
            new_skyline.append({"node_id": p_id, "dims": p_scores})
            skyline = new_skyline

    return skyline


def enhance_graph_with_semantic_links(
    graph: nx.Graph, embedder: TextEmbeddingProvider, x_percentile: float = 0.85
) -> nx.Graph:
    """

    Returns:
        nx.Graph: 添加了新的语义链接后的增强图。
    """
    enhanced_graph = graph.copy()
    nodes = list(graph.nodes())
    n_nodes = len(nodes)

    # 如果节点数不足以添加边，直接返回副本
    if len(nodes) <= 1:
        return enhanced_graph

    print(f"Enhancing graph with {len(nodes)} nodes...")

    try:
        node_embeddings = embedder.embed_texts(nodes)
    except Exception as e:
        print(f"Error during bulk embedding: {e}")
        return enhanced_graph

    # 计算所有节点对之间的余弦相似度矩阵
    # sim_matrix[i][j] 是 nodes[i] 和 nodes[j] 之间的相似度
    sim_matrix = cosine_similarity(node_embeddings)

    # 2.1 创建一个从节点名到矩阵索引的映射，用于高效查找
    node_to_idx = {node_name: i for i, node_name in enumerate(nodes)}
    # 2.2 遍历原始图的边，提取它们的相似度
    existing_edge_similarities = []
    for u, v in graph.edges():
        if u in node_to_idx and v in node_to_idx:
            idx_u = node_to_idx[u]
            idx_v = node_to_idx[v]
            similarity = sim_matrix[idx_u, idx_v]
            existing_edge_similarities.append(similarity)

    # 2.3 基于提取出的相似度计算阈值
    if existing_edge_similarities:
        scores_array = np.array(existing_edge_similarities)
        similarity_threshold = np.percentile(scores_array, x_percentile * 100)
    else:
        # 如果原始图没有边，无法计算基准，设置一个非常高的阈值
        # 这样就不会添加任何新边，是安全的默认行为
        similarity_threshold = 1.0

    log.info(f"Base similarities on {len(existing_edge_similarities)} existing edges.")
    log.info(
        f"Calculated similarity threshold at {x_percentile*100:.0f}th percentile: {similarity_threshold:.4f}"
    )

    if n_nodes > 0:
        # 平均度数 = 2 * |E| / |V|
        average_degree = (2 * graph.number_of_edges()) / n_nodes
        # 应用 ceiling 并确保最小值至少为1
        max_new_edges = max(1, math.ceil(average_degree))
    else:
        max_new_edges = 1  # 对于空图的安全默认值

    log.info(f"Original graph average degree: {average_degree:.2f}")
    log.info(f"Global limit for adding new edges per node set to: {max_new_edges}")

    for i, source_node in enumerate(nodes):
        new_edges_added_for_node = 0

        similarities = sorted(
            zip(sim_matrix[i], nodes), key=lambda x: x[0], reverse=True
        )

        for sim_score, target_node in similarities:
            if new_edges_added_for_node >= max_new_edges:
                break
            if source_node == target_node:
                continue

            # 条件1: 相似度高于全局阈值
            # 条件2: 边在 *原始* 图中不存在
            if sim_score > similarity_threshold and not graph.has_edge(
                source_node, target_node
            ):
                if not enhanced_graph.has_edge(source_node, target_node):
                    enhanced_graph.add_edge(
                        source_node, target_node, weight=float(sim_score)
                    )
                    new_edges_added_for_node += 1

    original_edges = graph.number_of_edges()
    new_edges = enhanced_graph.number_of_edges()
    log.info(
        f"Enhancement complete. Edges increased from {original_edges} to {new_edges}."
    )

    return enhanced_graph
