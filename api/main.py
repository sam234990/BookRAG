"""BookRAG FastAPI application entry point."""
from dotenv import load_dotenv
load_dotenv()  # load .env before any os.getenv / os.environ calls

import logging
import os
import uuid as _uuid
import json
from contextlib import asynccontextmanager

import yaml
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.db import mongodb as db
from api.dependencies import MONGO_URI, MONGO_DB_PREFIX, MONGO_SYSTEM_DB, THREAD_POOL
from api.routers import auth, documents, chat, tenants, entities


# ── Structured JSON logging ──────────────────────────────────────────────────

class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            obj["request_id"] = record.request_id
        if record.exc_info and record.exc_info[0] is not None:
            obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(obj, default=str)


_log_level = os.getenv("BOOKRAG_LOG_LEVEL", "INFO").upper()
_handler = logging.StreamHandler()
_handler.setFormatter(_JSONFormatter())
logging.root.handlers = [_handler]
logging.root.setLevel(getattr(logging, _log_level, logging.INFO))
log = logging.getLogger(__name__)


# ── Config validation ────────────────────────────────────────────────────────

_CONFIG_PATH = os.getenv("BOOKRAG_CONFIG_PATH", "config/gbc.yaml")
_CONFIG_REQUIRED_KEYS = {"llm", "vlm"}  # top-level sections that must exist


def _validate_config(path: str):
    """Validate YAML config at startup — fail fast on missing required sections."""
    if not os.path.isfile(path):
        raise RuntimeError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Config file is not a valid YAML mapping: {path}")
    missing = _CONFIG_REQUIRED_KEYS - raw.keys()
    if missing:
        raise RuntimeError(f"Config file '{path}' is missing required sections: {missing}")
    log.info(f"Config validated: {path}")


# ── Request-ID middleware ────────────────────────────────────────────────────

class _RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject an ``X-Request-ID`` header (echo or generate) and attach to log context."""

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(_uuid.uuid4())
        # Make request_id available in logging context (thread-local filter)
        _request_id_ctx.set(req_id)
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


import contextvars
_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class _RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get("-")  # type: ignore[attr-defined]
        return True


logging.root.addFilter(_RequestIDFilter())


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    log.info("BookRAG API starting up...")
    # Validate config
    _validate_config(_CONFIG_PATH)

    # Ensure upload and index directories exist
    os.makedirs(os.getenv("BOOKRAG_UPLOAD_DIR", "./uploads"), exist_ok=True)
    os.makedirs(os.getenv("BOOKRAG_INDEX_DIR", "./indices"), exist_ok=True)

    # Build MongoDB indexes and recover stale indexing for all known tenants
    tenant_ids: list[str] = []
    try:
        sdb = db.get_system_db(MONGO_URI, MONGO_SYSTEM_DB)
        tenant_ids = [t["tenant_id"] async for t in sdb["tenants"].find({}, {"tenant_id": 1})]
        await db.ensure_indexes(MONGO_URI, MONGO_SYSTEM_DB, MONGO_DB_PREFIX, tenant_ids)
    except Exception as exc:
        log.warning(f"MongoDB index creation skipped: {exc}")

    # Recover docs stuck in "indexing" status from a previous crash
    if tenant_ids:
        try:
            recovered = await db.recover_stale_indexing(MONGO_URI, MONGO_DB_PREFIX, tenant_ids)
            if recovered:
                log.warning(f"Recovered {recovered} stale indexing document(s) → status='error'")
        except Exception as exc:
            log.warning(f"Stale indexing recovery skipped: {exc}")

    yield

    # Graceful shutdown: finish running tasks, don't cancel them
    log.info("BookRAG API shutting down — draining thread pool...")
    THREAD_POOL.shutdown(wait=True, cancel_futures=False)
    await db.close_client()
    log.info("BookRAG API shut down cleanly.")


app = FastAPI(
    title="BookRAG API",
    description="Multi-tenant, multi-document chatbot powered by GBC-RAG",
    version="1.0.0",
    lifespan=lifespan,
)

# Request-ID middleware (must be added before CORS)
app.add_middleware(_RequestIDMiddleware)

# CORS — set BOOKRAG_CORS_ORIGINS to a comma-separated list of allowed origins
_cors_raw = os.getenv("BOOKRAG_CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
if "*" in _cors_origins:
    log.warning(
        "CORS allow_origins contains '*'. This is insecure with credentials=True. "
        "Set BOOKRAG_CORS_ORIGINS to explicit origins in production."
    )
    _cors_credentials = False
else:
    _cors_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

# Routers
app.include_router(auth.router)
app.include_router(tenants.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(entities.router)


@app.get("/health")
async def health():
    """Deep health check — verify MongoDB and FalkorDB connectivity."""
    checks: dict = {"service": "BookRAG API"}

    # MongoDB ping
    try:
        client = db.get_client(MONGO_URI)
        await client.admin.command("ping")
        checks["mongodb"] = "ok"
    except Exception as exc:
        checks["mongodb"] = f"error: {exc}"

    # FalkorDB ping (optional — only if host is configured)
    fdb_host = os.getenv("BOOKRAG_FALKORDB_HOST", "")
    if fdb_host:
        try:
            from api.dependencies import FALKORDB_HOST, FALKORDB_PORT, FALKORDB_PASSWORD
            import falkordb
            fdb = falkordb.FalkorDB(
                host=FALKORDB_HOST, port=FALKORDB_PORT,
                password=FALKORDB_PASSWORD or None,
            )
            fdb.connection.ping()
            checks["falkordb"] = "ok"
        except Exception as exc:
            checks["falkordb"] = f"error: {exc}"

    overall = "ok" if all(v == "ok" for k, v in checks.items() if k != "service") else "degraded"
    checks["status"] = overall
    return checks

