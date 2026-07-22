# Copyright (C) 2025-2026 Shu Wang
# SPDX-License-Identifier: Apache-2.0 OR AGPL-3.0-only

from .base_config import BaseRAGStrategyConfig
from typing import Literal
from pydantic import Field
from Core.configs.vdb_config import VDBConfig


class MMConfig(BaseRAGStrategyConfig):
    vdb_config: VDBConfig = Field(default_factory=VDBConfig)
    strategy: Literal["mmr"] = "mmr"
    topk: int = Field(
        default=5, description="The number of topk retrieval results for MMRAG."
    )
