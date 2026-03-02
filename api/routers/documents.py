"""Document management router: upload (multi-file), list, status, raw download, delete."""
import logging
import os
import shutil
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Query, Form
from fastapi.responses import FileResponse
import aiofiles

from api.models.requests import DocumentResponse, BatchUploadResponse
from api.db import mongodb as db
from api.dependencies import (
    MONGO_URI, MONGO_DB_PREFIX, UPLOAD_DIR, INDEX_SAVE_DIR,
    get_current_user, check_doc_access,
)
from api.services.indexing import run_indexing

log = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

CONFIG_PATH = os.getenv("BOOKRAG_CONFIG_PATH", "config/gbc.yaml")

# Max upload size in bytes — default 200 MB
_MAX_UPLOAD_BYTES = int(os.getenv("BOOKRAG_MAX_UPLOAD_MB", "200")) * 1024 * 1024


async def _save_and_register_file(
    file: UploadFile,
    tenant_id: str,
    user_id: str,
    tenant_upload_dir: str,
) -> dict:
    """Save one uploaded file to the tenant upload dir and return doc metadata.

    Raises ``HTTPException(413)`` if the file exceeds ``_MAX_UPLOAD_BYTES``.
    """
    doc_id = str(uuid.uuid4())
    # Preserve original filename; prefix with doc_id to avoid collisions
    safe_name = os.path.basename(file.filename)
    pdf_path = os.path.join(tenant_upload_dir, f"{doc_id}_{safe_name}")

    total_size = 0
    async with aiofiles.open(pdf_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            total_size += len(chunk)
            if total_size > _MAX_UPLOAD_BYTES:
                # Clean up partial file
                await out.close()
                try:
                    os.remove(pdf_path)
                except OSError:
                    pass
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum upload size of {_MAX_UPLOAD_BYTES // (1024*1024)} MB",
                )
            await out.write(chunk)

    from datetime import timezone
    now = datetime.now(timezone.utc)
    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "tenant_id": tenant_id,
        "uploaded_by": user_id,
        "pdf_path": pdf_path,
        "created_at": now,
        "status": "pending",
    }


@router.post("", status_code=202, response_model=BatchUploadResponse)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    document_date: Optional[str] = Form(
        default=None,
        description="Optional ISO-8601 date for ALL uploaded files (original authoring date). "
                    "Example: 2025-06-15 or 2025-06-15T10:30:00Z",
    ),
    document_lang: Optional[str] = Form(
        default=None,
        description="Optional ISO 639-1 language code (e.g. 'en', 'id') for ALL uploaded files. "
                    "Omit or set to 'auto' for automatic detection from extracted text.",
    ),
    current_user: dict = Depends(get_current_user),
):
    """Upload one or more PDFs and start background indexing for each.

    An optional ``document_date`` (ISO-8601) can be provided to indicate
    the original authoring/publishing date of the documents.  This date is
    used for temporal-awareness in cross-document RAG queries.

    An optional ``document_lang`` (ISO 639-1 code like ``en``, ``id``) can
    be provided to hint the document language for legal heading detection
    and text processing.  Omit for automatic detection.
    """
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]

    # Parse optional document_date
    parsed_doc_date: Optional[datetime] = None
    if document_date:
        try:
            parsed_doc_date = datetime.fromisoformat(document_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=422, detail="document_date must be a valid ISO-8601 date string")

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

        # Attach document_date if provided
        if parsed_doc_date:
            doc_data["document_date"] = parsed_doc_date

        # Attach document_lang if provided
        if document_lang:
            doc_data["document_lang"] = document_lang

        # Register document in MongoDB
        await db.create_document(MONGO_URI, MONGO_DB_PREFIX, tenant_id, doc_data)
        # Auto-grant uploader owner access
        await db.grant_permission(MONGO_URI, MONGO_DB_PREFIX, tenant_id, user_id, doc_data["doc_id"], "owner")
        # Enqueue background indexing
        background_tasks.add_task(
            run_indexing, tenant_id, doc_data["doc_id"], doc_data["pdf_path"], CONFIG_PATH,
            document_date=parsed_doc_date,
            document_lang=document_lang,
        )
        uploaded.append(DocumentResponse(
            doc_id=doc_data["doc_id"], filename=file.filename, status="pending",
            document_date=parsed_doc_date,
            document_lang=document_lang,
        ))

    return BatchUploadResponse(uploaded=uploaded, failed=failed)


@router.get("", response_model=List[DocumentResponse])
async def list_documents(
    limit: int = Query(default=50, ge=1, le=200, description="Max documents to return"),
    offset: int = Query(default=0, ge=0, description="Number of documents to skip"),
    current_user: dict = Depends(get_current_user),
):
    """List documents accessible to the current user, sorted by document_date descending."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    docs, _total = await db.list_documents(MONGO_URI, MONGO_DB_PREFIX, tenant_id, user_id, limit=limit, offset=offset)
    return [
        DocumentResponse(
            doc_id=d["doc_id"],
            filename=d.get("filename", ""),
            status=d.get("status", "unknown"),
            error=d.get("error"),
            created_at=d.get("created_at"),
            document_date=d.get("document_date"),
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
        created_at=doc.get("created_at"),
        document_date=doc.get("document_date"),
    )


@router.delete("/{doc_id}", status_code=204)
async def delete_document(doc_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a document and all associated indexes, VDB data, and FalkorDB graph.

    Requires the requesting user to be the document owner or an admin.
    """
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]

    # Only owner or admin can delete
    if current_user["role"] != "admin":
        perm = await db.get_permission(MONGO_URI, MONGO_DB_PREFIX, tenant_id, user_id, doc_id)
        if not perm or perm.get("role") != "owner":
            raise HTTPException(status_code=403, detail="Only document owners or admins can delete documents")

    doc = await db.get_document(MONGO_URI, MONGO_DB_PREFIX, tenant_id, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Clean up filesystem: uploaded PDF
    pdf_path = doc.get("pdf_path", "")
    if pdf_path and os.path.isfile(pdf_path):
        try:
            os.remove(pdf_path)
        except OSError:
            log.warning(f"Could not remove uploaded PDF: {pdf_path}")

    # Clean up filesystem: index directory
    index_dir = os.path.join(INDEX_SAVE_DIR, tenant_id, doc_id)
    if os.path.isdir(index_dir):
        try:
            shutil.rmtree(index_dir)
        except OSError:
            log.warning(f"Could not remove index directory: {index_dir}")

    # Clean up FalkorDB graph (best-effort)
    try:
        from api.dependencies import FALKORDB_HOST, FALKORDB_PORT, FALKORDB_PASSWORD
        if os.getenv("BOOKRAG_FALKORDB_HOST", ""):
            import falkordb
            fdb = falkordb.FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT, password=FALKORDB_PASSWORD or None)
            graph_name = f"bookrag:{tenant_id}:doc:{doc_id}"
            try:
                g = fdb.select_graph(graph_name)
                g.delete()
                log.info(f"Deleted FalkorDB graph '{graph_name}'")
            except Exception:
                pass  # Graph may not exist
    except Exception as exc:
        log.warning(f"FalkorDB cleanup skipped: {exc}")

    # Clean up MongoDB records
    await db.delete_document(MONGO_URI, MONGO_DB_PREFIX, tenant_id, doc_id)


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

