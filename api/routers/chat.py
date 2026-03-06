"""Chat router: query, session management."""
import logging
import os
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query

from api.models.requests import (
    ChatQueryRequest, ChatQueryResponse,
    SessionCreateRequest, SessionResponse,
    SessionMessagesResponse, MessageResponse,
    SessionListResponse, SessionListItem,
)
from api.db import mongodb as db
from api.dependencies import (
    MONGO_URI, MONGO_DB_PREFIX,
    get_current_user, filter_accessible_docs,
    rate_limit_query,
)
from api.services.chat import handle_query

log = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

CONFIG_PATH = os.getenv("BOOKRAG_CONFIG_PATH", "config/gbc.yaml")


@router.post("/query", response_model=ChatQueryResponse)
async def query(req: ChatQueryRequest, current_user: dict = Depends(rate_limit_query)):
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


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    limit: int = Query(default=50, ge=1, le=200, description="Max sessions to return"),
    offset: int = Query(default=0, ge=0, description="Number of sessions to skip"),
    current_user: dict = Depends(get_current_user),
):
    """List all chat sessions for the current user, newest first."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    sessions, total = await db.list_sessions(
        MONGO_URI, MONGO_DB_PREFIX, tenant_id, user_id, limit=limit, offset=offset
    )
    items = [
        SessionListItem(
            session_id=s["session_id"],
            created_at=s.get("created_at"),
            message_count=s.get("message_count", len(s.get("messages", []))),
            doc_ids=s.get("doc_ids", []),
        )
        for s in sessions
    ]
    return SessionListResponse(sessions=items, total=total)


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a chat session and all its messages."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    session = await db.get_session(MONGO_URI, MONGO_DB_PREFIX, tenant_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get("user_id") != user_id and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    await db.delete_session(MONGO_URI, MONGO_DB_PREFIX, tenant_id, session_id)


@router.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
async def get_messages(
    session_id: str,
    limit: int = Query(default=100, ge=1, le=500, description="Max messages to return"),
    offset: int = Query(default=0, ge=0, description="Number of messages to skip"),
    current_user: dict = Depends(get_current_user),
):
    """Retrieve messages in a session with pagination."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    session = await db.get_session(MONGO_URI, MONGO_DB_PREFIX, tenant_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get("user_id") != user_id and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    all_messages = session.get("messages", [])
    total = len(all_messages)
    paginated = all_messages[offset:offset + limit]
    messages = [
        MessageResponse(
            role=m["role"],
            content=m["content"],
            ts=str(m.get("ts", "")),
        )
        for m in paginated
    ]
    return SessionMessagesResponse(session_id=session_id, messages=messages, total=total)

