"""BookRAG FastAPI application entry point."""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.db import mongodb as db
from api.dependencies import MONGO_URI, MONGO_DB_PREFIX, MONGO_SYSTEM_DB
from api.routers import auth, documents, chat, tenants, entities

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    log.info("BookRAG API starting up...")
    # Ensure upload and index directories exist
    os.makedirs(os.getenv("BOOKRAG_UPLOAD_DIR", "./uploads"), exist_ok=True)
    os.makedirs(os.getenv("BOOKRAG_INDEX_DIR", "./indices"), exist_ok=True)

    # Build MongoDB indexes for all known tenants
    try:
        sdb = db.get_system_db(MONGO_URI, MONGO_SYSTEM_DB)
        tenant_ids = [t["tenant_id"] async for t in sdb["tenants"].find({}, {"tenant_id": 1})]
        await db.ensure_indexes(MONGO_URI, MONGO_SYSTEM_DB, MONGO_DB_PREFIX, tenant_ids)
    except Exception as exc:
        log.warning(f"MongoDB index creation skipped: {exc}")

    yield
    log.info("BookRAG API shutting down...")
    await db.close_client()


app = FastAPI(
    title="BookRAG API",
    description="Multi-tenant, multi-document chatbot powered by GBC-RAG",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — set BOOKRAG_CORS_ORIGINS to a comma-separated list of allowed origins
_cors_raw = os.getenv("BOOKRAG_CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
if "*" in _cors_origins:
    log.warning(
        "CORS allow_origins contains '*'. This is insecure with credentials=True. "
        "Set BOOKRAG_CORS_ORIGINS to explicit origins in production."
    )
    # Wildcard + credentials is rejected by browsers; fall back to no-credentials mode
    _cors_credentials = False
else:
    _cors_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Routers
app.include_router(auth.router)
app.include_router(tenants.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(entities.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "BookRAG API"}

