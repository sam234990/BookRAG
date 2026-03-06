from pydantic import BaseModel, Field


class EntityResolutionConfig(BaseModel):
    """Tenant/global canonical entity resolution settings."""

    enabled: bool = False
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    top_k: int = Field(default=1, ge=1, le=20)
    global_vdb_dir: str = "./indices"
    collection_name: str = "global_kg_collection"
    canonical_only: bool = False
    sync_to_global_graph: bool = False