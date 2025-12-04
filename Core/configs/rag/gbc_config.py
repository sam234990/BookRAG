from .base_config import BaseRAGStrategyConfig
from typing import Literal
from pydantic import Field

from Core.configs.rerank_config import RerankerConfig
from Core.configs.embedding_config import EmbeddingConfig


class GBCRAGConfig(BaseRAGStrategyConfig):
    """
    Configuration class for the GBC RAG (Graph-Based Contextual Retrieval Augmented Generation).
    This class defines the parameters required for initializing the GBC RAG agent.
    """

    strategy: Literal["gbc"] = "gbc"
    varient: Literal["standard", "wo_plan", "wo_selector", "wo_graph", "wo_text", "wo_er"] = Field(
        default="standard",
        description="The variant of the GBC RAG strategy to use. Options are 'standard', 'wo_plan', 'wo_selector', 'wo_graph', 'wo_text', and 'wo_er'.",
    )
    topk: int = Field(
        default=10,
        description="The number of top results to return from the graph-based retrieval.",
    )
    sim_threshold_e: float = Field(
        default=0.3,
        description="The similarity threshold for filtering retrieved results.",
    )
    select_depth: int = Field(
        default=2,
        description="The tree depth of section for LLM selection.",
    )
    x_percentile: float = Field(
        default=0.85,
        description="The percentile for selecting the top x% of edge similarity, used in Graph Augmentation.",
    )
    alpha: float = Field(
        default=0.5,
        description="PPR parameter.",
    )
    topk_ent: int = Field(
        default=5,
        description="The number of top entities to retrieve from the graph.",
    )

    max_retry: int = Field(
        default=3,
        description="The maximum number of retries for the LLM to generate a valid response.",
    )
    reranker_config: RerankerConfig = Field(
        default_factory=RerankerConfig,
    )
    mm_reranker_config: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
    )
