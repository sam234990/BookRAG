# Core/configs/rag/__init__.py

from .traverse_config import TraverseRAGConfig
from .gbc_config import GBCRAGConfig
from .mm_config import MMConfig
from .graph_config import GraphRAGConfig
from .gbc_vanilla_config import GBCVanillaConfig
from .vanilla_config import VanillaConfig

ALL_STRATEGY_CONFIGS = (
    TraverseRAGConfig,
    GBCRAGConfig,
    MMConfig,
    GraphRAGConfig,
    VanillaConfig,
    GBCVanillaConfig,
)
