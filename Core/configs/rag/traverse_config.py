# Copyright (C) 2025-2026 Shu Wang
# SPDX-License-Identifier: Apache-2.0 OR AGPL-3.0-only

# Core/configs/rag/traverse_config.py
from pydantic import Field
from typing import Literal
from .base_config import BaseRAGStrategyConfig

class TraverseRAGConfig(BaseRAGStrategyConfig):
    strategy: Literal["traverse"] = "traverse"
    max_depth: int = Field(
        default=5,
        description="The maximum depth for the document tree traversal."
    )
