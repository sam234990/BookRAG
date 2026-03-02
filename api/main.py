"""BookRAG FastAPI application entry point."""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.db import mongodb as db
from api.dependencies import MONGO_URI
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
    yield
    log.info("BookRAG API shutting down...")
    await db.close_client()


app = FastAPI(
    title="BookRAG API",
    description="Multi-tenant, multi-document chatbot powered by GBC-RAG",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — adjust origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("BOOKRAG_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

