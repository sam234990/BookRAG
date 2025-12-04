import yaml
from Core.configs.mineru_config import MinerU
from Core.configs.llm_config import LLMConfig
from Core.configs.tree_config import TreeConfig
from Core.configs.graph_config import GraphConfig
from Core.configs.vlm_config import VLMConfig
from Core.configs.rag_config import RAGConfig
from Core.configs.vdb_config import VDBConfig
from pydantic import BaseModel, Field
from typing import Optional


class SystemConfig(BaseModel):
    """
    Top-level application configuration model.
    Pydantic will automatically handle nested validation and instantiation.
    """

    # LLM Configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    mineru: MinerU = Field(default_factory=MinerU)

    # Index Configurations
    tree: TreeConfig = Field(default_factory=TreeConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    vdb: VDBConfig = Field(default_factory=VDBConfig)

    # Other Index selection
    index_type: Optional[str] = "gbc"  # Options: "gbc", "tree", "vanilla", "bm25", "raptor", "pdf_vanilla"

    rag_force_reprocess: Optional[bool] = False

    # RAG Configurations
    rag: RAGConfig = Field(default_factory=RAGConfig)

    # Paths
    pdf_path: Optional[str] = "/home/wangshu/multimodal/GBC-RAG/test/double_paper.pdf"
    save_path: Optional[str] = "/home/wangshu/multimodal/GBC-RAG/test/tree_index"

    # # 新增: 专门用于存放评估结果的根目录
    # evaluation_output_path: Optional[str] = Field(
    #     default="/home/wangshu/multimodal/GBC-RAG/test/tree_index/evaluation_results",
    #     description="Root directory to save evaluation results."
    # )


def load_system_config(path: str = "../configs/default.yaml") -> SystemConfig:
    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    if "rag" in raw_config:
        rag_data = raw_config["rag"]
        raw_config["rag"] = {"strategy_config": rag_data}

    cfg = SystemConfig(**raw_config)
    return cfg
