from .base_config import BaseRAGStrategyConfig
from typing import Literal
from pydantic import Field
from Core.configs.vdb_config import VDBConfig


class GBCVanillaConfig(BaseRAGStrategyConfig):
    strategy: Literal["gbcvanilla"] = "gbcvanilla"
    tree_vdb_config: VDBConfig = Field(default_factory=VDBConfig)
    topk: int = Field(
        default=10,
        description="The number of top results to return from the tree-vdb retrieval.",
    )
    
    graph_vdb_config: VDBConfig = Field(default_factory=VDBConfig)
    topk_ent: int = Field(
        default=5,
        description="The number of top entities to retrieve from the graph.",
    )
