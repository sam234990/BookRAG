"""Document management router: upload (multi-file), list, status, raw download."""
import logging
import os
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
import aiofiles

from api.models.requests import DocumentResponse, BatchUploadResponse
from api.db import mongodb as db
from api.dependencies import (
    MONGO_URI, MONGO_DB_PREFIX, UPLOAD_DIR,
    get_current_user, check_doc_access,
)
from api.services.indexing import run_indexing

log = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

CONFIG_PATH = os.getenv("BOOKRAG_CONFIG_PATH", "config/gbc.yaml")


async def _save_and_register_file(
    file: UploadFile,
    tenant_id: str,
    user_id: str,
    tenant_upload_dir: str,
) -> dict:
    """Save one uploaded file to the tenant upload dir and return doc metadata."""
    doc_id = str(uuid.uuid4())
    # Preserve original filename; prefix with doc_id to avoid collisions
    safe_name = os.path.basename(file.filename)
    pdf_path = os.path.join(tenant_upload_dir, f"{doc_id}_{safe_name}")

    async with aiofiles.open(pdf_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            await out.write(chunk)

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "tenant_id": tenant_id,
        "uploaded_by": user_id,
        "pdf_path": pdf_path,
    }


@router.post("", status_code=202, response_model=BatchUploadResponse)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Upload one or more PDFs and start background indexing for each."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]

    tenant_upload_dir = os.path.join(UPLOAD_DIR, tenant_id)
    os.makedirs(tenant_upload_dir, exist_ok=True)

    uploaded: List[DocumentResponse] = []
    failed: List[dict] = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            failed.append({"filename": file.filename, "error": "Only PDF files are supported"})
            continue
        try:
            doc_data = await _save_and_register_file(file, tenant_id, user_id, tenant_upload_dir)
        except Exception as exc:
            log.error(f"Failed to save file '{file.filename}': {exc}")
            failed.append({"filename": file.filename, "error": str(exc)})
            continue

        # Register document in MongoDB
        await db.create_document(MONGO_URI, MONGO_DB_PREFIX, tenant_id, doc_data)
        # Auto-grant uploader owner access
        await db.grant_permission(MONGO_URI, MONGO_DB_PREFIX, tenant_id, user_id, doc_data["doc_id"], "owner")
        # Enqueue background indexing
        background_tasks.add_task(
            run_indexing, tenant_id, doc_data["doc_id"], doc_data["pdf_path"], CONFIG_PATH
        )
        uploaded.append(DocumentResponse(doc_id=doc_data["doc_id"], filename=file.filename, status="pending"))

    return BatchUploadResponse(uploaded=uploaded, failed=failed)


@router.get("", response_model=List[DocumentResponse])
async def list_documents(current_user: dict = Depends(get_current_user)):
    """List all documents accessible to the current user."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    docs = await db.list_documents(MONGO_URI, MONGO_DB_PREFIX, tenant_id, user_id)
    return [
        DocumentResponse(
            doc_id=d["doc_id"],
            filename=d.get("filename", ""),
            status=d.get("status", "unknown"),
            error=d.get("error"),
        )
        for d in docs
    ]


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document_status(doc_id: str, current_user: dict = Depends(get_current_user)):
    """Get indexing status for a specific document."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]

    if not await check_doc_access(user_id, tenant_id, doc_id):
        raise HTTPException(status_code=403, detail="Access denied to this document")

    doc = await db.get_document(MONGO_URI, MONGO_DB_PREFIX, tenant_id, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse(
        doc_id=doc["doc_id"],
        filename=doc.get("filename", ""),
        status=doc.get("status", "unknown"),
        error=doc.get("error"),
    )


@router.get("/{doc_id}/raw")
async def download_raw_document(doc_id: str, current_user: dict = Depends(get_current_user)):
    """Stream back the original uploaded PDF file."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]

    if not await check_doc_access(user_id, tenant_id, doc_id):
        raise HTTPException(status_code=403, detail="Access denied to this document")

    raw_path = await db.get_document_raw_path(MONGO_URI, MONGO_DB_PREFIX, tenant_id, doc_id)
    if not raw_path or not os.path.isfile(raw_path):
        raise HTTPException(status_code=404, detail="Raw document file not found")

    # Prevent path-traversal: resolved path must be inside UPLOAD_DIR
    resolved = os.path.realpath(raw_path)
    upload_root = os.path.realpath(UPLOAD_DIR)
    if not resolved.startswith(upload_root + os.sep) and resolved != upload_root:
        log.warning(f"Path traversal blocked: {raw_path} resolved to {resolved}")
        raise HTTPException(status_code=403, detail="Access denied")

    filename = os.path.basename(raw_path)
    return FileResponse(
        path=raw_path,
        media_type="application/pdf",
        filename=filename,
    )

