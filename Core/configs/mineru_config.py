from dataclasses import dataclass

@dataclass
class MinerU:
    backend: str
    method: str
    lang:str
    server_url: str = "http://127.0.0.1:30000"

    def __post_init__(self):
        if self.backend not in [
            "vlm-sglang-client",
            "vlm-transformers",
            "vlm-sglang-engine",
            "pipeline",
        ]:
            raise ValueError(f"Unsupported backend: {self.backend}")
        self.method = "auto" if self.backend == "pipeline" else "vlm"
