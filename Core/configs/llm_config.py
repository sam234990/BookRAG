import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()  # ensure .env is loaded when used outside the API server

@dataclass
class LLMConfig:
    model_name: str = "Qwen/Qwen3-8B-AWQ"
    api_key: str = "openai"
    api_base: str = "http://localhost:8003/v1"
    temperature: float = 0.1
    max_tokens: int = 5000
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    backend: str = "openai"
    max_workers: int = 8

    def __post_init__(self):
        if self.backend not in ["openai", "ollama"]:
            raise ValueError(f"Unsupported backend: {self.backend}")
        # Allow api_key to be resolved from environment variable
        if not self.api_key or self.api_key in ("env", "ENV"):
            env_key = os.environ.get("CHAT_API_KEY", "") or os.environ.get("DASHSCOPE_API_KEY", "")
            if env_key:
                self.api_key = env_key
            else:
                raise ValueError(
                    "LLM api_key is empty/env but neither CHAT_API_KEY nor "
                    "DASHSCOPE_API_KEY environment variable is set."
                )
