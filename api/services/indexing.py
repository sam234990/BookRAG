"""Background indexing service: PDF → GBC Index."""
import asyncio
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

from api.db import mongodb as db
from api.dependencies import MONGO_URI, MONGO_DB_PREFIX, INDEX_SAVE_DIR

log = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2)


def _build_index_sync(pdf_path: str, save_path: str, tenant_id: str, doc_id: str, config_path: str):
    """Synchronous index build — runs in a thread pool."""
    from Core.configs.system_config import load_system_config
    from Core.configs.falkordb_config import FalkorDBConfig
    from Core.pipelines.doc_tree_builder import construct_GBC_index

    cfg = load_system_config(config_path)
    cfg.pdf_path = pdf_path
    cfg.save_path = save_path
    cfg.tenant_id = tenant_id
    cfg.doc_id = doc_id
    # FalkorDB will be used if BOOKRAG_FALKORDB_HOST is set
    fdb_host = os.getenv("BOOKRAG_FALKORDB_HOST", "")
    if fdb_host:
        from api.dependencies import FALKORDB_HOST, FALKORDB_PORT, FALKORDB_PASSWORD
        cfg.falkordb = FalkorDBConfig(
            host=FALKORDB_HOST, port=FALKORDB_PORT, password=FALKORDB_PASSWORD
        )
    construct_GBC_index(cfg)


async def run_indexing(
    tenant_id: str,
    doc_id: str,
    pdf_path: str,
    config_path: str,
):
    """Async wrapper: update status in MongoDB before/after indexing."""
    save_path = os.path.join(INDEX_SAVE_DIR, tenant_id, doc_id)
    os.makedirs(save_path, exist_ok=True)

    await db.update_document_status(MONGO_URI, MONGO_DB_PREFIX, tenant_id, doc_id, "indexing")
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _executor,
            _build_index_sync,
            pdf_path, save_path, tenant_id, doc_id, config_path,
        )
        await db.update_document_status(MONGO_URI, MONGO_DB_PREFIX, tenant_id, doc_id, "ready")
        log.info(f"Indexing complete for doc '{doc_id}' in tenant '{tenant_id}'")
        # Phase 3: Run entity resolution to merge into global graph
        try:
            from api.services.entity_resolution import run_entity_resolution
            await run_entity_resolution(tenant_id, doc_id, config_path)
        except Exception as er_err:
            log.warning(f"Entity resolution skipped (non-fatal): {er_err}")
    except Exception as e:
        log.error(f"Indexing failed for doc '{doc_id}': {e}", exc_info=True)
        await db.update_document_status(MONGO_URI, MONGO_DB_PREFIX, tenant_id, doc_id, "error", str(e))

