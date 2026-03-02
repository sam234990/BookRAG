"""Pydantic request and response models for the BookRAG API."""
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
    access_token: str
    token_type: str = "bearer"


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
    doc_id: str
    filename: str
    status: str  # pending | indexing | ready | error
    error: Optional[str] = None


class BatchUploadResponse(BaseModel):
    uploaded: List["DocumentResponse"]
    failed: List[dict] = Field(default_factory=list)  # {"filename": ..., "error": ...}


class PermissionGrantRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=_SHORT_STR)
    doc_id: str = Field(..., min_length=1, max_length=_SHORT_STR)
    role: str = Field(default="reader", max_length=32)


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=_QUERY_MAX)
    session_id: Optional[str] = Field(default=None, max_length=_SHORT_STR)
    doc_ids: Optional[List[str]] = None  # restrict to specific docs; None = all accessible
    cross_doc: bool = False  # use global cross-document graph


class ChatQueryResponse(BaseModel):
    answer: str
    session_id: str
    doc_ids_used: List[str] = Field(default_factory=list)
    rewritten_query: Optional[str] = None  # set when history was used to rewrite the query


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

