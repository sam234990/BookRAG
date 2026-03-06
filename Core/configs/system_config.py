import os

import yaml
from Core.configs.mineru_config import MinerU
from Core.configs.docling_config import DoclingConfig
from Core.configs.entity_resolution_config import EntityResolutionConfig
from Core.configs.llm_config import LLMConfig
from Core.configs.tree_config import TreeConfig
from Core.configs.graph_config import GraphConfig
from Core.configs.ontology_config import OntologyConfig
from Core.configs.vlm_config import VLMConfig
from Core.configs.rag_config import RAGConfig
from Core.configs.vdb_config import VDBConfig
from Core.configs.falkordb_config import FalkorDBConfig
from Core.configs.mongodb_config import MongoDBConfig
from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime


class SystemConfig(BaseModel):
    """
    Top-level application configuration model.
    Pydantic will automatically handle nested validation and instantiation.
    """

    # LLM Configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    mineru: MinerU = Field(default_factory=MinerU)

    # Parser selection: "mineru" (default) or "docling"
    parser: Optional[str] = "mineru"
    docling: Optional[DoclingConfig] = Field(default_factory=DoclingConfig)

    # Index Configurations
    tree: TreeConfig = Field(default_factory=TreeConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    vdb: VDBConfig = Field(default_factory=VDBConfig)
    ontology: OntologyConfig = Field(default_factory=OntologyConfig)
    entity_resolution: EntityResolutionConfig = Field(default_factory=EntityResolutionConfig)

    # Other Index selection
    index_type: Optional[str] = "gbc"  # Options: "gbc", "tree", "vanilla", "bm25", "raptor", "pdf_vanilla"

    rag_force_reprocess: Optional[bool] = False

    # RAG Configurations
    rag: RAGConfig = Field(default_factory=RAGConfig)

    # Paths
    pdf_path: Optional[str] = "/home/wangshu/multimodal/GBC-RAG/test/double_paper.pdf"
    save_path: Optional[str] = "/home/wangshu/multimodal/GBC-RAG/test/tree_index"

    # Multi-tenant identifiers (optional for backward compatibility)
    tenant_id: Optional[str] = None
    doc_id: Optional[str] = None

    # Document language hint (ISO 639-1).  Used by the legal-heading
    # detector, incomplete-paragraph heuristics, and other language-aware
    # pipeline stages.  Set to "auto" (default) for automatic detection
    # from extracted text, or an explicit code like "en" or "id".
    document_lang: Optional[str] = Field(
        default="auto",
        description="ISO 639-1 language code of the document content, or "
                    "'auto' for automatic detection. "
                    "Supported: auto, en, id, de, fr, es, pt, it, nl, th, zh, ja, ko, ar.",
    )

    # Document temporal metadata (optional, for recency-aware RAG)
    document_date: Optional[datetime] = Field(
        default=None,
        description="Original authoring/publishing date of the document. "
                    "Used for temporal awareness in cross-document RAG queries.",
    )

    # Database configurations
    falkordb: Any = Field(default_factory=FalkorDBConfig)
    mongodb: Any = Field(default_factory=MongoDBConfig)


def load_system_config(path: str = "../configs/default.yaml") -> SystemConfig:
    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    if "rag" in raw_config:
        rag_data = raw_config["rag"]
        raw_config["rag"] = {"strategy_config": rag_data}

    ontology_data = raw_config.get("ontology")
    if isinstance(ontology_data, dict) and ontology_data.get("path"):
        ontology_path = ontology_data["path"]
        if not os.path.isabs(ontology_path):
            ontology_data["path"] = os.path.abspath(
                os.path.join(os.path.dirname(path), ontology_path)
            )

    entity_resolution_data = raw_config.get("entity_resolution")
    if isinstance(entity_resolution_data, dict) and entity_resolution_data.get("global_vdb_dir"):
        global_vdb_dir = entity_resolution_data["global_vdb_dir"]
        if not os.path.isabs(global_vdb_dir):
            entity_resolution_data["global_vdb_dir"] = os.path.abspath(
                os.path.join(os.path.dirname(path), global_vdb_dir)
            )

    cfg = SystemConfig(**raw_config)
    return cfg
