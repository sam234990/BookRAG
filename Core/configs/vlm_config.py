import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()  # ensure .env is loaded when used outside the API server


@dataclass
class VLMConfig:
    backend: str = "gpt"  # "qwen", "gpt", "ollama"
    model_name: str = "Qwen/Qwen3.5-35B-A3B-AWQ"
    max_tokens: int = 6000
    temperature: float = 0.1
    api_key: str = "openai"
    api_base: str = "http://localhost:8003/v1"

    def __post_init__(self):
        # Allow api_key to be resolved from environment variable
        if not self.api_key or self.api_key in ("env", "ENV"):
            env_key = os.environ.get("VL_API_KEY", "") or os.environ.get("DASHSCOPE_API_KEY", "")
            if env_key:
                self.api_key = env_key
            else:
                raise ValueError(
                    "VLM api_key is empty/env but neither VL_API_KEY nor "
                    "DASHSCOPE_API_KEY environment variable is not set."
                )
