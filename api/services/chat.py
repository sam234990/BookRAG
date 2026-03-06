"""Chat service: single-doc and cross-doc query routing."""
import asyncio
import logging
import os
import threading
import time
import uuid
from functools import lru_cache
from typing import List, Optional

from api.db import mongodb as db
from api.dependencies import (
    MONGO_URI, MONGO_DB_PREFIX, INDEX_SAVE_DIR,
    FALKORDB_HOST, FALKORDB_PORT, FALKORDB_USERNAME, FALKORDB_PASSWORD,
    THREAD_POOL,
)

log = logging.getLogger(__name__)
_executor = THREAD_POOL

# ── Object caching ───────────────────────────────────────────────────────────
# LLM/VLM are API client wrappers — one instance per config is sufficient.
# GBC indexes are heavier (tree + graph + VDB) — cached per document with TTL.

_GBC_CACHE_TTL = int(os.getenv("BOOKRAG_GBC_CACHE_TTL", "600"))  # seconds
_GBC_CACHE_MAX = int(os.getenv("BOOKRAG_GBC_CACHE_MAX", "20"))   # max cached indexes


@lru_cache(maxsize=4)
def _get_system_config(config_path: str):
    """Cached system config loader — reloads only when config_path changes."""
    from Core.configs.system_config import load_system_config
    return load_system_config(config_path)


@lru_cache(maxsize=4)
def _get_llm(config_path: str):
    """Singleton LLM instance per config file."""
    from Core.provider.llm import LLM
    cfg = _get_system_config(config_path)
    return LLM(cfg.llm)


@lru_cache(maxsize=4)
def _get_vlm(config_path: str):
    """Singleton VLM instance per config file."""
    from Core.provider.vlm import VLM
    cfg = _get_system_config(config_path)
    if hasattr(cfg, "vlm") and cfg.vlm:
        return VLM(cfg.vlm)
    return None


class _GBCCache:
    """TTL-bounded LRU cache for per-document GBC indexes."""

    def __init__(self, max_size: int = 20, ttl: int = 600):
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, object]] = {}  # key → (access_time, gbc)
        self._max_size = max_size
        self._ttl = ttl

    def get(self, key: str):
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            ts, gbc = entry
            if time.monotonic() - ts > self._ttl:
                del self._cache[key]
                return None
            self._cache[key] = (time.monotonic(), gbc)
            return gbc

    def put(self, key: str, gbc):
        with self._lock:
            self._cache[key] = (time.monotonic(), gbc)
            # Evict oldest if over capacity
            if len(self._cache) > self._max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]

    def invalidate(self, key: str):
        with self._lock:
            self._cache.pop(key, None)


_gbc_cache = _GBCCache(max_size=_GBC_CACHE_MAX, ttl=_GBC_CACHE_TTL)


def _get_gbc_index(tenant_id: str, doc_id: str, config_path: str):
    """Load or return cached GBC index for a specific document."""
    from Core.configs.falkordb_config import FalkorDBConfig
    from Core.Index.GBCIndex import GBC

    cache_key = f"{tenant_id}:{doc_id}"
    cached = _gbc_cache.get(cache_key)
    if cached is not None:
        log.debug(f"GBC cache hit: {cache_key}")
        return cached

    log.info(f"GBC cache miss: {cache_key} — loading from disk")
    cfg = _get_system_config(config_path)
    # Create a copy-like config with tenant/doc specifics
    cfg.tenant_id = tenant_id
    cfg.doc_id = doc_id
    cfg.save_path = os.path.join(INDEX_SAVE_DIR, tenant_id, doc_id)

    fdb_host = os.getenv("BOOKRAG_FALKORDB_HOST", "")
    if fdb_host:
        cfg.falkordb = FalkorDBConfig(
            host=FALKORDB_HOST, port=FALKORDB_PORT,
            username=FALKORDB_USERNAME, password=FALKORDB_PASSWORD,
        )

    gbc_index = GBC.load_gbc_index(cfg)
    _gbc_cache.put(cache_key, gbc_index)
    return gbc_index

# ── History relevance constants (tunable via env) ─────────────────────────────
_RECENT_TURNS    = int(os.getenv("BOOKRAG_RECENT_TURNS",    "3"))   # always-include last N pairs
_MAX_OLD_MSGS    = int(os.getenv("BOOKRAG_MAX_OLD_MSGS",    "4"))   # max older msgs to keep
_JACCARD_THRESH  = float(os.getenv("BOOKRAG_JACCARD_THRESH", "0.15")) # token-overlap threshold


# ── History filtering helpers ─────────────────────────────────────────────────

def _jaccard_similarity(a: str, b: str) -> float:
    """Token-overlap Jaccard similarity — zero-dependency relevance heuristic."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _filter_relevant_history(
    query: str,
    messages: List[dict],
    recent_turns: int = _RECENT_TURNS,
    max_old: int = _MAX_OLD_MSGS,
    threshold: float = _JACCARD_THRESH,
) -> List[dict]:
    """Return the subset of prior messages relevant to *query*.

    Two-tier strategy:
    1. **Recency** — always keep the last ``recent_turns`` user+assistant pairs.
    2. **Relevance** — for older messages, keep those whose content has
       Jaccard token-overlap >= *threshold* with the query (capped at *max_old*).
    """
    if not messages:
        return []

    recent_cutoff = recent_turns * 2          # 2 messages per turn (user + assistant)
    recent  = messages[-recent_cutoff:]
    older   = messages[:-recent_cutoff] if len(messages) > recent_cutoff else []

    relevant_older = [
        m for m in older
        if _jaccard_similarity(query, m.get("content", "")) >= threshold
    ][-max_old:]

    return relevant_older + recent


# ── Query rewriting (sync, runs inside thread pool) ───────────────────────────

def _rewrite_query_sync(query: str, history: List[dict], config_path: str) -> str:
    """Use the LLM to rewrite *query* as a self-contained question.

    Resolves pronouns and implicit references using the conversation history.
    If the LLM fails or the result is empty, the original query is returned unchanged.
    """
    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in history
    )
    prompt = (
        "You are a query rewriter for a document question-answering system.\n"
        "Given the conversation history below and the user's latest question, "
        "rewrite the question as a single, self-contained question that can be "
        "understood without any prior context. Resolve all pronouns, coreferences, "
        "and implicit references to named entities mentioned earlier in the conversation. "
        "If the question is already fully self-contained, return it unchanged.\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"Latest question: {query}\n\n"
        "Rewritten question (return ONLY the rewritten question, no preamble or explanation):"
    )
    try:
        llm = _get_llm(config_path)
        rewritten = llm.get_completion(prompt).strip()
        if rewritten:
            log.info(f"Query rewritten: '{query}' → '{rewritten}'")
            return rewritten
    except Exception as exc:
        log.warning(f"Query rewrite failed ({exc}); using original query.")
    return query


def _query_single_doc_sync(
    query: str, tenant_id: str, doc_id: str, config_path: str, lang: str = "en"
) -> str:
    """Run GBC RAG query against a single document (sync, for thread pool).

    *query* should already be a self-contained, rewritten query when conversation
    history is present (see :func:`_rewrite_query_sync`).
    """
    from Core.rag.gbc_rag import GBCRAG
    from Core.configs.rag.gbc_config import GBCRAGConfig

    gbc_index = _get_gbc_index(tenant_id, doc_id, config_path)
    llm = _get_llm(config_path)
    vlm = _get_vlm(config_path)
    rag_cfg = GBCRAGConfig()
    rag = GBCRAG(llm=llm, vlm=vlm, config=rag_cfg, gbc_index=gbc_index, lang=lang)
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
    """Route query to appropriate retrieval mode and store in session.

    Memory strategy
    ---------------
    1. Load existing session messages from MongoDB (before appending the new one).
    2. Filter to relevant history using :func:`_filter_relevant_history`
       (two-tier: recency + Jaccard token-overlap).
    3. If non-empty history, rewrite *query* into a self-contained standalone
       question via the LLM (:func:`_rewrite_query_sync`).
    4. Pass the (possibly rewritten) query to the RAG pipeline.
    """
    loop = asyncio.get_event_loop()

    # ── Session bootstrap ──────────────────────────────────────────────────────
    if not session_id:
        session_id = str(uuid.uuid4())
        await db.create_session(MONGO_URI, MONGO_DB_PREFIX, tenant_id, {
            "session_id": session_id,
            "user_id": user_id,
            "doc_ids": doc_ids,
            "messages": [],
        })
        prior_messages: List[dict] = []
    else:
        # Load history BEFORE appending the current user message
        session = await db.get_session(MONGO_URI, MONGO_DB_PREFIX, tenant_id, session_id)
        prior_messages = session.get("messages", []) if session else []

    # ── History filtering + query rewriting ───────────────────────────────────
    relevant_history = _filter_relevant_history(query, prior_messages)
    if relevant_history:
        log.debug(
            f"Session {session_id}: {len(relevant_history)} relevant history messages "
            f"out of {len(prior_messages)} total — rewriting query."
        )
        effective_query = await loop.run_in_executor(
            _executor, _rewrite_query_sync, query, relevant_history, config_path
        )
    else:
        effective_query = query

    # ── Persist user message (original, for readability) ──────────────────────
    await db.append_message(MONGO_URI, MONGO_DB_PREFIX, tenant_id, session_id,
                            {"role": "user", "content": query})

    # ── RAG retrieval ─────────────────────────────────────────────────────────

    # ── Fetch per-doc metadata (dates + languages) — best-effort ──────────
    doc_dates: dict[str, str] = {}
    doc_langs: dict[str, str] = {}
    try:
        for did in doc_ids[:5]:
            doc_record = await db.get_document(MONGO_URI, MONGO_DB_PREFIX, tenant_id, did)
            if doc_record:
                ddate = doc_record.get("document_date") or doc_record.get("created_at")
                if ddate:
                    doc_dates[did] = str(ddate)[:10]  # YYYY-MM-DD
                dlang = doc_record.get("document_lang")
                if dlang and dlang != "auto":
                    doc_langs[did] = dlang
    except Exception:
        pass  # Non-fatal: metadata is best-effort

    if cross_doc or len(doc_ids) > 1:
        # Parallel per-doc queries, answers synthesised into one response
        target_docs = doc_ids[:5]   # cap to avoid GPU overload
        answers = await asyncio.gather(*[
            loop.run_in_executor(
                _executor, _query_single_doc_sync,
                effective_query, tenant_id, did, config_path,
                doc_langs.get(did, "en"),
            )
            for did in target_docs
        ])

        # Build answer with temporal + language context
        parts = []
        for did, ans in zip(target_docs, answers):
            date_str = f" (dated {doc_dates[did]})" if did in doc_dates else ""
            lang_str = f" [lang: {doc_langs[did]}]" if did in doc_langs else ""
            parts.append(f"[Document: {did}{date_str}{lang_str}]\n{ans}")

        # Prepend contextual notes when metadata is present
        notes = []
        if doc_dates:
            notes.append(
                "NOTE: The answers below come from multiple documents with different dates. "
                "When documents contain contradictory or overlapping information, "
                "prefer the information from the more recently dated document."
            )
        unique_langs = set(doc_langs.values())
        if len(unique_langs) > 1:
            notes.append(
                "NOTE: The answers below come from documents in different languages. "
                "Each answer is in its document's language; synthesise accordingly."
            )
        if notes:
            answer = "\n\n".join(notes) + "\n\n" + "\n\n---\n\n".join(parts)
        else:
            answer = "\n\n---\n\n".join(parts)
    else:
        doc_id = doc_ids[0] if doc_ids else None
        if not doc_id:
            answer = "No accessible documents found for your query."
        else:
            lang = doc_langs.get(doc_id, "en")
            answer = await loop.run_in_executor(
                _executor, _query_single_doc_sync,
                effective_query, tenant_id, doc_id, config_path, lang,
            )

    # ── Persist assistant message ──────────────────────────────────────────────
    await db.append_message(MONGO_URI, MONGO_DB_PREFIX, tenant_id, session_id,
                            {"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "session_id": session_id,
        "doc_ids_used": doc_ids,
        "rewritten_query": effective_query if effective_query != query else None,
    }

