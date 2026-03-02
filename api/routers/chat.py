"""Chat router: query, session management."""
import logging
import os
from typing import List
from fastapi import APIRouter, Depends, HTTPException

from api.models.requests import (
    ChatQueryRequest, ChatQueryResponse,
    SessionCreateRequest, SessionResponse,
    SessionMessagesResponse, MessageResponse,
)
from api.db import mongodb as db
from api.dependencies import (
    MONGO_URI, MONGO_DB_PREFIX,
    get_current_user, filter_accessible_docs,
)
from api.services.chat import handle_query

log = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

CONFIG_PATH = os.getenv("BOOKRAG_CONFIG_PATH", "config/gbc.yaml")


@router.post("/query", response_model=ChatQueryResponse)
async def query(req: ChatQueryRequest, current_user: dict = Depends(get_current_user)):
    """Submit a query. Automatically filters to accessible documents."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]

    accessible_docs = await filter_accessible_docs(user_id, tenant_id, req.doc_ids)
    if not accessible_docs:
        raise HTTPException(status_code=403, detail="No accessible documents for this query")

    result = await handle_query(
        query=req.query,
        tenant_id=tenant_id,
        user_id=user_id,
        doc_ids=accessible_docs,
        session_id=req.session_id,
        config_path=CONFIG_PATH,
        cross_doc=req.cross_doc,
    )
    return ChatQueryResponse(**result)


@router.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(req: SessionCreateRequest, current_user: dict = Depends(get_current_user)):
    """Create a new chat session."""
    import uuid
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    session_id = str(uuid.uuid4())
    accessible_docs = await filter_accessible_docs(user_id, tenant_id, req.doc_ids)
    await db.create_session(MONGO_URI, MONGO_DB_PREFIX, tenant_id, {
        "session_id": session_id,
        "user_id": user_id,
        "doc_ids": accessible_docs,
        "messages": [],
    })
    return SessionResponse(session_id=session_id)


@router.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
async def get_messages(session_id: str, current_user: dict = Depends(get_current_user)):
    """Retrieve all messages in a session."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    session = await db.get_session(MONGO_URI, MONGO_DB_PREFIX, tenant_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get("user_id") != user_id and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    messages = [
        MessageResponse(
            role=m["role"],
            content=m["content"],
            ts=str(m.get("ts", "")),
        )
        for m in session.get("messages", [])
    ]
    return SessionMessagesResponse(session_id=session_id, messages=messages)

