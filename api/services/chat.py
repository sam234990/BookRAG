"""Chat service: single-doc and cross-doc query routing."""
import asyncio
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from api.db import mongodb as db
from api.dependencies import (
    MONGO_URI, MONGO_DB_PREFIX, INDEX_SAVE_DIR,
    FALKORDB_HOST, FALKORDB_PORT, FALKORDB_PASSWORD,
)

log = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=4)


def _query_single_doc_sync(query: str, tenant_id: str, doc_id: str, config_path: str) -> str:
    """Run GBC RAG query against a single document (sync, for thread pool)."""
    from Core.configs.system_config import load_system_config
    from Core.configs.falkordb_config import FalkorDBConfig
    from Core.Index.GBCIndex import GBC
    from Core.rag.gbc_rag import GBCRAG
    from Core.provider.llm import LLM
    from Core.provider.vlm import VLM
    from Core.configs.rag.gbc_config import GBCRAGConfig

    cfg = load_system_config(config_path)
    cfg.tenant_id = tenant_id
    cfg.doc_id = doc_id
    cfg.save_path = os.path.join(INDEX_SAVE_DIR, tenant_id, doc_id)

    fdb_host = os.getenv("BOOKRAG_FALKORDB_HOST", "")
    if fdb_host:
        cfg.falkordb = FalkorDBConfig(host=FALKORDB_HOST, port=FALKORDB_PORT, password=FALKORDB_PASSWORD)

    gbc_index = GBC.load_gbc_index(cfg)
    llm = LLM(cfg.llm)
    vlm = VLM(cfg.vlm) if hasattr(cfg, "vlm") else None
    rag_cfg = GBCRAGConfig()
    rag = GBCRAG(llm=llm, vlm=vlm, config=rag_cfg, gbc_index=gbc_index)
    result = rag.get_GBC_info(query)
    return result if isinstance(result, str) else str(result)


async def handle_query(
    query: str,
    tenant_id: str,
    user_id: str,
    doc_ids: List[str],
    session_id: Optional[str],
    config_path: str,
    cross_doc: bool = False,
) -> dict:
    """Route query to appropriate retrieval mode and store in session."""
    # Ensure session exists
    if not session_id:
        session_id = str(uuid.uuid4())
        await db.create_session(MONGO_URI, MONGO_DB_PREFIX, tenant_id, {
            "session_id": session_id,
            "user_id": user_id,
            "doc_ids": doc_ids,
            "messages": [],
        })

    # Store user message
    await db.append_message(MONGO_URI, MONGO_DB_PREFIX, tenant_id, session_id,
                            {"role": "user", "content": query})

    loop = asyncio.get_event_loop()

    if cross_doc or len(doc_ids) > 1:
        # Phase 3: parallel per-doc queries, answers synthesised into one response
        # Cap at 5 docs to prevent overloading GPU services
        target_docs = doc_ids[:5]
        answers = await asyncio.gather(*[
            loop.run_in_executor(_executor, _query_single_doc_sync, query, tenant_id, did, config_path)
            for did in target_docs
        ])
        answer = "\n\n---\n\n".join(f"[Document: {did}]\n{ans}" for did, ans in zip(target_docs, answers))
    else:
        doc_id = doc_ids[0] if doc_ids else None
        if not doc_id:
            answer = "No accessible documents found for your query."
        else:
            answer = await loop.run_in_executor(
                _executor, _query_single_doc_sync, query, tenant_id, doc_id, config_path
            )

    # Store assistant message
    await db.append_message(MONGO_URI, MONGO_DB_PREFIX, tenant_id, session_id,
                            {"role": "assistant", "content": answer})

    return {"answer": answer, "session_id": session_id, "doc_ids_used": doc_ids}

