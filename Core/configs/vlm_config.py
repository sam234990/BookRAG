# Copyright (C) 2025-2026 Shu Wang
# SPDX-License-Identifier: Apache-2.0 OR AGPL-3.0-only

from dataclasses import dataclass


@dataclass
class VLMConfig:
    backend: str = "ollama"  # "qwen", "gpt", "ollama"
    model_name: str = "qwen2.5vl:6k"
    max_tokens: int = 6000
    temperature: float = 0.7
    api_key: str = "None"
    api_base: str = "http://localhost:11434"
