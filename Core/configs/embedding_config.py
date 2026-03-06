import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


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
        # Resolve 'env' placeholder → read from DASHSCOPE_API_KEY environment variable
        if self.api_key == "env":
            self.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
            if not self.api_key:
                raise ValueError(
                    "EmbeddingConfig.api_key is 'env' but DASHSCOPE_API_KEY "
                    "environment variable is not set."
                )
