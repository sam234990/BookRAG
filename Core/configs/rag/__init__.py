# Copyright (C) 2025-2026 Shu Wang
# SPDX-License-Identifier: Apache-2.0 OR AGPL-3.0-only

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
