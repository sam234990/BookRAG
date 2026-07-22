# Copyright (C) 2025-2026 Shu Wang
# SPDX-License-Identifier: Apache-2.0 OR AGPL-3.0-only

from pydantic import BaseModel, Field
from typing import Union
from Core.configs.rag import ALL_STRATEGY_CONFIGS

StrategyConfig = Union[*ALL_STRATEGY_CONFIGS]


class RAGConfig(BaseModel):
    strategy_config: StrategyConfig = Field(..., discriminator="strategy")
