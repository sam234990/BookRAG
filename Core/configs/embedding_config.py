# Copyright (C) 2025-2026 Shu Wang
# SPDX-License-Identifier: Apache-2.0 OR AGPL-3.0-only

from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    type: str = "text"
    backend: str = "local"
    api_key: str = "ollama"
    api_base: str = "http://localhost:11434"
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    max_length: int = 8192
    device: str = "cuda:2"
    
    
    def __post_init__(self):
        if self.backend not in ["local", "ollama", "openai"]:
            raise ValueError(f"Unsupported backend: {self.backend}")
