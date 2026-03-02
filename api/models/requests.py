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


class BatchUploadResponse(BaseModel):
    uploaded: List["DocumentResponse"]
    failed: List[dict] = Field(default_factory=list)  # {"filename": ..., "error": ...}


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

