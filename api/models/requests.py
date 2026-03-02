"""Pydantic request and response models for the BookRAG API."""
from typing import List, Optional
from pydantic import BaseModel, Field


# ── Auth ──────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    password: str
    tenant_id: str


class LoginRequest(BaseModel):
    username: str
    password: str
    tenant_id: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Tenant ────────────────────────────────────────────────────────────────────

class TenantCreateRequest(BaseModel):
    tenant_id: str
    name: str
    description: Optional[str] = ""


class TenantResponse(BaseModel):
    tenant_id: str
    name: str
    description: Optional[str] = ""


# ── Document ──────────────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    status: str  # pending | indexing | ready | error
    error: Optional[str] = None


class PermissionGrantRequest(BaseModel):
    user_id: str
    doc_id: str
    role: str = "reader"


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatQueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    doc_ids: Optional[List[str]] = None  # restrict to specific docs; None = all accessible
    cross_doc: bool = False  # use global cross-document graph


class ChatQueryResponse(BaseModel):
    answer: str
    session_id: str
    doc_ids_used: List[str] = Field(default_factory=list)


class SessionCreateRequest(BaseModel):
    doc_ids: Optional[List[str]] = None


class SessionResponse(BaseModel):
    session_id: str


class MessageResponse(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    ts: Optional[str] = None


class SessionMessagesResponse(BaseModel):
    session_id: str
    messages: List[MessageResponse]

