from dataclasses import dataclass


@dataclass
class RerankerConfig:
    model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    max_length: int = 8192
    device: str = "cuda:2"
    backend: str = "local"  # Options: 'local', 'vllm'
    api_base: str = "http://localhost:8011/v1"
