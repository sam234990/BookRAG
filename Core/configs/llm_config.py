from dataclasses import dataclass

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
