from Core.Index.Graph import Entity
from Core.configs.entity_resolution_config import EntityResolutionConfig


def should_resolve_entity_globally(
    entity: Entity, resolution_cfg: EntityResolutionConfig
) -> bool:
    if not resolution_cfg.canonical_only:
        return True
    return bool(entity.canonical_id or entity.entity_role == "canonical")


def build_global_entity_metadata(entity: Entity, tenant_id: str, doc_id: str) -> dict:
    metadata = entity.to_vdb_metadata()
    metadata["tenant_id"] = tenant_id or ""
    metadata["doc_id"] = doc_id or ""
    metadata["canonical_name"] = entity.entity_name
    return metadata