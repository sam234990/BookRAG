from typing import List, Tuple

from Core.provider.llm import LLM
from Core.provider.embedding import TextEmbeddingProvider
from Core.utils.utils import TextProcessor, num_tokens
import logging

logger = logging.getLogger(__name__)


def GMM_cluster(
    embeddings,
    threshold: float = 0.1,
    random_state: int = 0,
    dim: int = 5,
    reg_covar: float = 1e-6,
) -> Tuple[List, int]:
    """
    对给定的嵌入向量进行GMM聚类。

    Args:
        embeddings (np.ndarray): 输入的嵌入向量数组。
        threshold (float): 判定样本属于一个聚类的概率阈值。
        random_state (int): 随机种子，保证结果可复现。
        dim (int): UMAP降维后的目标维度。
        reg_covar (float): GMM的正则化参数，用于增加协方差矩阵的稳定性。

    Returns:
        Tuple[List, int]: 一个元组，包含每个样本的聚类标签列表和最优的聚类数量。
    """
    import numpy as np
    import umap
    from sklearn.mixture import GaussianMixture
    import random

    random.seed(224)
    
    # 0. 处理微小输入的边缘情况
    if len(embeddings) == 0:
        return [], 0

    # 1. 提升数据精度，增加数值稳定性
    embeddings = embeddings.astype(np.float64)

    # 如果样本数量过少，无法进行有意义的聚类，直接返回单个聚类
    if len(embeddings) <= dim + 1:
        # 每个样本都属于聚类0
        labels = [np.array([0]) for _ in embeddings]
        return labels, 1

    # 2. UMAP降维
    # UMAP的n_components必须小于样本数
    n_components_umap = min(dim, len(embeddings) - 1)
    # 确保n_components至少为1
    if n_components_umap < 1:
        labels = [np.array([0]) for _ in embeddings]
        return labels, 1

    reduced_embeddings_global = umap.UMAP(
        n_neighbors=max(2, int((len(embeddings) - 1) ** 0.5)),  # n_neighbors >= 2
        n_components=n_components_umap,
        metric="cosine",
    ).fit_transform(embeddings)

    # 3. 确定一个更保守和安全的聚类数量搜索范围
    if len(reduced_embeddings_global) > 5000:
        max_clusters = len(reduced_embeddings_global) // 100
        n_clusters = np.arange(max(2, max_clusters - 1), max_clusters + 1)
    else:
        # 确保每个簇平均至少有 (比如) 10 个点，且最大不超过50
        max_clusters = min(50, len(reduced_embeddings_global) // 10)
        # 保证max_clusters至少为2，这样arange才能生成有效范围
        max_clusters = max(2, max_clusters)
        n_clusters = np.arange(1, max_clusters)

    if len(n_clusters) == 0:
        n_clusters = np.array([1])

    # 4. 通过BIC寻找最优聚类数，并加入健壮性处理
    bics = []
    valid_n_clusters = []
    for n in n_clusters:
        try:
            gm = GaussianMixture(
                n_components=n,
                random_state=random_state,
                reg_covar=reg_covar,  # 核心修改：增加正则化项
            )
            gm.fit(reduced_embeddings_global)
            bics.append(gm.bic(reduced_embeddings_global))
            valid_n_clusters.append(n)
        except ValueError as e:
            # 如果即使加了reg_covar仍然失败，则跳过这个n，而不是让程序崩溃
            logger.warning(
                f"GMM fitting failed for n_components={n}. Skipping. Error: {e}"
            )
            continue

    # 5. 处理所有聚类尝试都失败的边缘情况
    if not valid_n_clusters:
        logger.warning(
            "GMM fitting failed for all attempted n_components. Returning 1 cluster as a fallback."
        )
        labels = [np.array([0]) for _ in embeddings]
        return labels, 1

    # 确定最优聚类数量
    optimal_clusters = valid_n_clusters[np.argmin(bics)]

    # 6. 使用最优参数进行最终的GMM拟合
    final_gm = GaussianMixture(
        n_components=optimal_clusters,
        random_state=random_state,
        reg_covar=reg_covar,  # 同样需要正则化
    )
    final_gm.fit(reduced_embeddings_global)

    # 7. 预测每个样本属于各个聚类的概率，并根据阈值生成标签
    probs = final_gm.predict_proba(reduced_embeddings_global)
    labels = [np.where(prob > threshold)[0] for prob in probs]

    return labels, optimal_clusters


def get_embedding(texts: List[str], embedder: TextEmbeddingProvider):
    import numpy as np

    embeddings = embedder.embed_texts(texts)
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    return embeddings


def get_summary_prompt(cluster_texts: List[str], max_tokens=3000):
    SUMMARIZE = """You are a helpful assistant. Write a summary of the following, including as many key details as possible: {context}:"""

    context = "\n".join(cluster_texts)
    prompt = SUMMARIZE.format(context=context)
    chunks = TextProcessor.split_text_into_chunks(text=prompt, max_length=max_tokens)
    return chunks[0]


def batch_generate_summary(prompts: List[str], llm: LLM) -> List[str]:
    summaries = llm.batch_get_completion(prompts, json_response=False)
    if isinstance(summaries, str):
        summaries = [summaries]

    res_summaries = []
    for summary in summaries:
        if num_tokens(summary) > 1200:
            chunks = TextProcessor.split_text_into_chunks(text=summary, max_length=1200)
            res_summaries.append(chunks[0])
        else:
            res_summaries.append(summary)
    return res_summaries


def cluster_one_layer(
    input_texts: List[str], embedder: TextEmbeddingProvider, llm: LLM
) -> List[str]:

    embeddings = get_embedding(input_texts, embedder)
    labels, n = GMM_cluster(embeddings)

    print(f"Clustered into {n} groups.")

    summaries = []
    for cluster_id in range(n):
        # cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_indices = [i for i, label in enumerate(labels) if cluster_id in label]

        cluster_texts = [input_texts[i] for i in cluster_indices]
        summary_prompt = get_summary_prompt(cluster_texts)
        summaries.append(summary_prompt)
    summaries = batch_generate_summary(summaries, llm)
    return summaries


def get_meta_data(texts: List[str], depth: int, base_num: int) -> List[dict]:

    source = "document" if depth == 0 else f"depth_{depth}"
    meta_datas = [
        {"source": source, "chunk_id": base_num + i} for i in range(len(texts))
    ]
    return meta_datas


def raptor_tree(
    chunks: List[str], embedder: TextEmbeddingProvider, llm: LLM, max_depth=20
):
    current_texts = chunks
    tree_text = []
    meta_data = []
    base_num = 0

    # add the original chunks as depth 0
    tree_text.extend(current_texts)
    meta_data.extend(get_meta_data(current_texts, depth=0, base_num=base_num))
    base_num += len(current_texts)

    for depth in range(max_depth):
        if len(current_texts) <= 5:
            break

        print(f"Clustering depth {depth+1} with {len(current_texts)} texts")
        summaries = cluster_one_layer(current_texts, embedder, llm)
        tree_text.extend(summaries)
        current_texts = summaries
        meta_data.extend(
            get_meta_data(current_texts, depth=depth + 1, base_num=base_num)
        )
        base_num += len(current_texts)

    return tree_text, meta_data
