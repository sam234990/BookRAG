"""Pydantic request and response models for the BookRAG API."""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

# ── Reusable length constraints ──────────────────────────────────────────────
_SHORT_STR = 128        # usernames, tenant_ids, role names
_PASSWORD_MIN = 8
_PASSWORD_MAX = 128
_QUERY_MAX = 10_000     # max characters for a chat query


# ── Auth ──────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=_SHORT_STR)
    password: str = Field(..., min_length=_PASSWORD_MIN, max_length=_PASSWORD_MAX)
    tenant_id: str = Field(..., min_length=1, max_length=_SHORT_STR)

    @field_validator("password")
    @classmethod
    def password_complexity(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=_SHORT_STR)
    password: str = Field(..., min_length=1, max_length=_PASSWORD_MAX)
    tenant_id: str = Field(..., min_length=1, max_length=_SHORT_STR)


class TokenResponse(BaseModel):
    access_token: str = Field(..., description="Short-lived JWT access token (default 60 min)")
    refresh_token: str = Field(..., description="Long-lived refresh token for rotation (default 7 days)")
    token_type: str = Field(default="bearer", description="OAuth2 token type")


class RefreshRequest(BaseModel):
    refresh_token: str


# ── Tenant ────────────────────────────────────────────────────────────────────

class TenantCreateRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1, max_length=_SHORT_STR)
    name: str = Field(..., min_length=1, max_length=256)
    description: Optional[str] = Field(default="", max_length=1000)


class TenantResponse(BaseModel):
    tenant_id: str
    name: str
    description: Optional[str] = ""


# ── Document ──────────────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    doc_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Indexing status: pending | indexing | ready | error")
    error: Optional[str] = Field(default=None, description="Error message if status is 'error'")
    created_at: Optional[datetime] = Field(default=None, description="Upload timestamp (UTC)")
    document_date: Optional[datetime] = Field(
        default=None,
        description="User-provided original authoring/publishing date of the document. "
                    "Used for temporal awareness in cross-document RAG.",
    )
    document_lang: Optional[str] = Field(
        default=None,
        description="ISO 639-1 language code (e.g. 'en', 'id') or 'auto' for auto-detection. "
                    "Used for legal heading detection and language-aware text processing.",
    )


class BatchUploadResponse(BaseModel):
    uploaded: List["DocumentResponse"] = Field(..., description="Successfully uploaded documents")
    failed: List[dict] = Field(default_factory=list, description="Files that failed: [{filename, error}]")


class PermissionGrantRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=_SHORT_STR)
    doc_id: str = Field(..., min_length=1, max_length=_SHORT_STR)
    role: str = Field(default="reader", max_length=32)


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=_QUERY_MAX, description="User question")
    session_id: Optional[str] = Field(default=None, max_length=_SHORT_STR, description="Existing session ID for history-aware queries")
    doc_ids: Optional[List[str]] = Field(default=None, description="Restrict to specific docs; None = all accessible")
    cross_doc: bool = Field(default=False, description="Use cross-document retrieval mode")


class ChatQueryResponse(BaseModel):
    answer: str = Field(..., description="LLM-generated answer")
    session_id: str = Field(..., description="Session ID (created if not provided)")
    doc_ids_used: List[str] = Field(default_factory=list, description="Document IDs used for retrieval")
    rewritten_query: Optional[str] = Field(default=None, description="Rewritten query when history was used")


class SessionCreateRequest(BaseModel):
    doc_ids: Optional[List[str]] = None


class SessionResponse(BaseModel):
    session_id: str


class SessionListItem(BaseModel):
    session_id: str
    created_at: Optional[datetime] = None
    message_count: int = 0
    doc_ids: List[str] = Field(default_factory=list)


class SessionListResponse(BaseModel):
    sessions: List[SessionListItem]
    total: int


class MessageResponse(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    ts: Optional[str] = None


class SessionMessagesResponse(BaseModel):
    session_id: str
    messages: List[MessageResponse]
    total: int = Field(0, description="Total messages in session (before pagination)")


# ── Entity Management ─────────────────────────────────────────────────────────

class EntityRef(BaseModel):
    entity_name: str
    entity_type: str


class EntityInfo(BaseModel):
    entity_name: str
    entity_type: str
    description: str
    source_ids: List[int]
    node_name: str


class EntityListResponse(BaseModel):
    entities: List[EntityInfo]
    total: int


class RenameEntityRequest(BaseModel):
    entity_name: str
    entity_type: str
    new_entity_name: str
    new_entity_type: str = ""  # empty → keep same type
    new_description: Optional[str] = None  # None → keep existing description


class MergeEntitiesRequest(BaseModel):
    source_entities: List[EntityRef]           # entities to merge (≥ 2)
    canonical_entity_name: str
    canonical_entity_type: str
    canonical_description: str = ""


class NewEntitySpec(BaseModel):
    entity_name: str
    entity_type: str
    description: str = ""
    source_ids: List[int] = Field(default_factory=list)


class SplitEntityRequest(BaseModel):
    entity_name: str
    entity_type: str
    new_entities: List[NewEntitySpec]          # ≥ 2 new entities
    edge_mode: str = "duplicate"               # "duplicate" | "none"


class MergeSuggestion(BaseModel):
    entity_a: EntityRef
    entity_b: EntityRef
    score: float                               # 0.0 – 1.0
    method: str                               # "string_similarity" | "embedding_similarity"


class SuggestMergesResponse(BaseModel):
    suggestions: List[MergeSuggestion]


class EntityOperationResponse(BaseModel):
    success: bool
    message: str
    entities: List[EntityInfo] = Field(default_factory=list)

