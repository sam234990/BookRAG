"""Background indexing service: PDF → GBC Index."""
import asyncio
import logging
import os
import shutil

from api.db import mongodb as db
from api.dependencies import MONGO_URI, MONGO_DB_PREFIX, INDEX_SAVE_DIR, THREAD_POOL

log = logging.getLogger(__name__)
_executor = THREAD_POOL


def _build_index_sync(
    pdf_path: str, save_path: str, tenant_id: str, doc_id: str,
    config_path: str, document_date=None, document_lang=None,
):
    """Synchronous index build — runs in a thread pool."""
    from Core.configs.system_config import load_system_config
    from Core.configs.falkordb_config import FalkorDBConfig
    from Core.construct_index import construct_gbc_index

    cfg = load_system_config(config_path)
    cfg.pdf_path = pdf_path
    cfg.save_path = save_path
    cfg.tenant_id = tenant_id
    cfg.doc_id = doc_id
    # Propagate document_date into the config for temporal awareness
    if document_date is not None:
        cfg.document_date = document_date
    # Propagate document_lang into the config for language-aware processing
    if document_lang is not None:
        cfg.document_lang = document_lang
    # FalkorDB will be used if BOOKRAG_FALKORDB_HOST is set
    fdb_host = os.getenv("BOOKRAG_FALKORDB_HOST", "")
    if fdb_host:
        from api.dependencies import FALKORDB_HOST, FALKORDB_PORT, FALKORDB_USERNAME, FALKORDB_PASSWORD
        cfg.falkordb = FalkorDBConfig(
            host=FALKORDB_HOST, port=FALKORDB_PORT,
            username=FALKORDB_USERNAME, password=FALKORDB_PASSWORD,
        )
    construct_gbc_index(cfg)


async def run_indexing(
    tenant_id: str,
    doc_id: str,
    pdf_path: str,
    config_path: str,
    document_date=None,
    document_lang=None,
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
            pdf_path, save_path, tenant_id, doc_id, config_path, document_date,
            document_lang,
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

