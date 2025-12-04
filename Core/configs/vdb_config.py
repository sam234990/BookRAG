from dataclasses import dataclass, field

from Core.configs.embedding_config import EmbeddingConfig


@dataclass
class VDBConfig:
    mm_embedding: bool = True
    vdb_dir_name: str = "./chroma_db"
    collection_name: str = "default_collection"
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    force_rebuild: bool = True
