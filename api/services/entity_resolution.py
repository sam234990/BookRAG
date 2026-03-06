"""
Phase 3: Cross-document entity resolution pipeline.

After a document is indexed, this service:
1. Loads document-level graph entities.
2. Searches the tenant-global VDB for cosine-similar canonical candidates.
3. Reuses an existing tenant-global canonical when the similarity gate is met.
4. Persists unmatched entities plus ontology metadata into the tenant-global stores.
"""
import asyncio
import logging
import os

from api.dependencies import THREAD_POOL

log = logging.getLogger(__name__)
_executor = THREAD_POOL


def _resolve_entities_sync(
    tenant_id: str,
    doc_id: str,
    config_path: str,
):
    """
    Synchronous entity resolution — runs in a thread pool after indexing.

    Steps:
      1. Load per-doc GBC index to get all new entities.
      2. Open (or create) global ChromaDB VDB for the tenant.
      3. For each new entity: search global VDB, LLM-verify top match if score > threshold.
      4. If verified merge: MERGE in global FalkorDB graph + update global VDB.
      5. If no match: add as new canonical entity in global VDB + global graph.
    """
    from Core.configs.system_config import load_system_config
    from Core.Index.GBCIndex import GBC
    from Core.provider.vdb import VectorStore
    from Core.utils.entity_resolution_utils import (
        build_global_entity_metadata,
        should_resolve_entity_globally,
    )
    from api.dependencies import INDEX_SAVE_DIR

    cfg = load_system_config(config_path)
    resolution_cfg = cfg.entity_resolution
    if not resolution_cfg.enabled:
        log.info("Entity resolution disabled by config; skipping Phase 3 sync.")
        return

    cfg.tenant_id = tenant_id
    cfg.doc_id = doc_id
    cfg.save_path = os.path.join(INDEX_SAVE_DIR, tenant_id, doc_id)

    falkordb_cfg = None
    if resolution_cfg.sync_to_global_graph and getattr(cfg.falkordb, "host", ""):
        falkordb_cfg = cfg.falkordb

    gbc = GBC.load_gbc_index(cfg)
    graph = gbc.GraphIndex
    embedder = gbc.embedder

    # Open global VDB for tenant
    global_vdb_path = os.path.join(resolution_cfg.global_vdb_dir, tenant_id, "global_vdb")
    global_vdb = VectorStore(
        db_path=global_vdb_path,
        embedding_model=embedder,
        collection_name=resolution_cfg.collection_name,
    )

    nodes = graph.get_all_nodes()
    new_canonical_texts = []
    new_canonical_meta = []

    for node_name in nodes:
        entity = graph.get_entity_by_node_name(node_name)
        if not should_resolve_entity_globally(entity, resolution_cfg):
            continue

        # Search global VDB for similar entity
        hits = global_vdb.search(node_name, top_k=resolution_cfg.top_k)
        merged = False
        if hits and hits[0]["distance"] < (1.0 - resolution_cfg.similarity_threshold):
            # Cosine distance is 1 - similarity; low distance = high similarity
            canonical_name = hits[0]["content"]
            log.info(
                f"Entity '{entity.entity_name}' similar to canonical '{canonical_name}' "
                f"(dist={hits[0]['distance']:.3f}). Merging."
            )
            merged = True

        if not merged:
            new_canonical_texts.append(node_name)
            new_canonical_meta.append(
                build_global_entity_metadata(entity, tenant_id=tenant_id, doc_id=doc_id)
            )

    if new_canonical_texts:
        global_vdb.add_texts(texts=new_canonical_texts, metadatas=new_canonical_meta)
        log.info(f"Added {len(new_canonical_texts)} new canonical entities to global VDB for tenant '{tenant_id}'.")

    # Push doc graph to global FalkorDB graph (idempotent MERGE)
    if falkordb_cfg:
        graph.save_to_global_graph(falkordb_cfg, tenant_id)
        log.info(f"Global FalkorDB graph updated for tenant '{tenant_id}', doc '{doc_id}'.")


async def run_entity_resolution(tenant_id: str, doc_id: str, config_path: str):
    """Async entry point for entity resolution — call after indexing completes."""
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            _executor, _resolve_entities_sync, tenant_id, doc_id, config_path
        )
    except Exception as e:
        log.error(f"Entity resolution failed for doc '{doc_id}': {e}", exc_info=True)

