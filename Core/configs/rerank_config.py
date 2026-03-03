import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class RerankerConfig:
    model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    max_length: int = 8192
    device: str = "cuda:2"
    backend: str = "local"  # Options: 'local', 'vllm', 'jina'
    api_base: str = "http://localhost:8011/v1"
    api_key: str = ""

    def __post_init__(self):
        if self.backend not in ["local", "vllm", "jina"]:
            raise ValueError(f"Unsupported reranker backend: {self.backend}")
        # Resolve 'env' placeholder → read from JINA_API_KEY environment variable
        if self.api_key == "env":
            self.api_key = os.environ.get("JINA_API_KEY", "")
            if not self.api_key:
                raise ValueError(
                    "RerankerConfig.api_key is 'env' but JINA_API_KEY "
                    "environment variable is not set."
                )
