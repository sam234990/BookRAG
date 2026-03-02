from dataclasses import dataclass


@dataclass
class VLMConfig:
    backend: str = "gpt"  # "qwen", "gpt", "ollama"
    model_name: str = "Qwen/Qwen3.5-35B-A3B-AWQ"
    max_tokens: int = 6000
    temperature: float = 0.1
    api_key: str = "openai"
    api_base: str = "http://localhost:8003/v1"
