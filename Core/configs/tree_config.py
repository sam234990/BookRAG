from dataclasses import dataclass

@dataclass
class TreeConfig:
    node_keywords: bool = True
    node_summary: bool = False
    use_vlm: bool = False
    