"""Document management router: upload, list, status."""
import logging
import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
import aiofiles

from api.models.requests import DocumentResponse
from api.db import mongodb as db
from api.dependencies import (
    MONGO_URI, MONGO_DB_PREFIX, UPLOAD_DIR,
    get_current_user,
)
from api.services.indexing import run_indexing

log = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

CONFIG_PATH = os.getenv("BOOKRAG_CONFIG_PATH", "config/gbc.yaml")


@router.post("", status_code=202, response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Upload a PDF and start background indexing."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    doc_id = str(uuid.uuid4())
    tenant_upload_dir = os.path.join(UPLOAD_DIR, tenant_id)
    os.makedirs(tenant_upload_dir, exist_ok=True)
    pdf_path = os.path.join(tenant_upload_dir, f"{doc_id}.pdf")

    # Save uploaded file
    async with aiofiles.open(pdf_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            await out.write(chunk)

    # Register document in MongoDB
    doc_data = {
        "doc_id": doc_id,
        "filename": file.filename,
        "tenant_id": tenant_id,
        "uploaded_by": user_id,
        "pdf_path": pdf_path,
    }
    await db.create_document(MONGO_URI, MONGO_DB_PREFIX, tenant_id, doc_data)

    # Auto-grant uploader read access
    await db.grant_permission(MONGO_URI, MONGO_DB_PREFIX, tenant_id, user_id, doc_id, "owner")

    # Start background indexing
    background_tasks.add_task(run_indexing, tenant_id, doc_id, pdf_path, CONFIG_PATH)

    return DocumentResponse(doc_id=doc_id, filename=file.filename, status="pending")


@router.get("", response_model=list[DocumentResponse])
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

    # Permission check
    from api.dependencies import check_doc_access
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

