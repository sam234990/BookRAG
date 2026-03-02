"""Entity management router: list, rename, merge, split, suggest-merges."""
import logging
import os

from fastapi import APIRouter, Depends, HTTPException

from api.models.requests import (
    EntityListResponse, EntityInfo,
    EntityOperationResponse,
    RenameEntityRequest,
    MergeEntitiesRequest,
    SplitEntityRequest,
    SuggestMergesResponse, MergeSuggestion, EntityRef,
)
from api.dependencies import get_current_user, check_doc_access
import api.services.entity_editor as svc

log = logging.getLogger(__name__)
router = APIRouter(prefix="/entities", tags=["entities"])

CONFIG_PATH = os.getenv("BOOKRAG_CONFIG_PATH", "config/gbc.yaml")


async def _require_access(tenant_id: str, user_id: str, doc_id: str):
    if not await check_doc_access(user_id, tenant_id, doc_id):
        raise HTTPException(status_code=403, detail="Access denied to this document")


# ── List ──────────────────────────────────────────────────────────────────────

@router.get("/{doc_id}", response_model=EntityListResponse)
async def list_entities(doc_id: str, current_user: dict = Depends(get_current_user)):
    """Return all NER entities for the given document."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    await _require_access(tenant_id, user_id, doc_id)

    try:
        entities = await svc.list_entities(tenant_id, doc_id, CONFIG_PATH)
    except Exception as exc:
        log.exception(f"list_entities failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return EntityListResponse(
        entities=[EntityInfo(**e) for e in entities],
        total=len(entities),
    )


# ── Rename ────────────────────────────────────────────────────────────────────

@router.patch("/{doc_id}/rename", response_model=EntityOperationResponse)
async def rename_entity(
    doc_id: str,
    req: RenameEntityRequest,
    current_user: dict = Depends(get_current_user),
):
    """Rename an entity node (name and/or type)."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    await _require_access(tenant_id, user_id, doc_id)

    try:
        updated = await svc.rename_entity(
            tenant_id=tenant_id, doc_id=doc_id, config_path=CONFIG_PATH,
            entity_name=req.entity_name, entity_type=req.entity_type,
            new_entity_name=req.new_entity_name,
            new_entity_type=req.new_entity_type,
            new_description=req.new_description,
            user_id=user_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        log.exception(f"rename_entity failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return EntityOperationResponse(
        success=True,
        message=f"Renamed '{req.entity_name}' → '{req.new_entity_name}'",
        entities=[EntityInfo(**e) for e in updated],
    )


# ── Merge ─────────────────────────────────────────────────────────────────────

@router.post("/{doc_id}/merge", response_model=EntityOperationResponse)
async def merge_entities(
    doc_id: str,
    req: MergeEntitiesRequest,
    current_user: dict = Depends(get_current_user),
):
    """Merge two or more entities into a single canonical entity."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    await _require_access(tenant_id, user_id, doc_id)

    if len(req.source_entities) < 2:
        raise HTTPException(status_code=422, detail="Provide at least 2 source_entities to merge")

    try:
        updated = await svc.merge_entities(
            tenant_id=tenant_id, doc_id=doc_id, config_path=CONFIG_PATH,
            source_entities=[e.model_dump() for e in req.source_entities],
            canonical_name=req.canonical_entity_name,
            canonical_type=req.canonical_entity_type,
            canonical_desc=req.canonical_description,
            user_id=user_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        log.exception(f"merge_entities failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return EntityOperationResponse(
        success=True,
        message=f"Merged {len(req.source_entities)} entities → '{req.canonical_entity_name}'",
        entities=[EntityInfo(**e) for e in updated],
    )


# ── Split ─────────────────────────────────────────────────────────────────────

@router.post("/{doc_id}/split", response_model=EntityOperationResponse)
async def split_entity(
    doc_id: str,
    req: SplitEntityRequest,
    current_user: dict = Depends(get_current_user),
):
    """Split one entity into two or more new entities."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    await _require_access(tenant_id, user_id, doc_id)

    if len(req.new_entities) < 2:
        raise HTTPException(status_code=422, detail="Provide at least 2 new_entities for a split")
    if req.edge_mode not in ("duplicate", "none"):
        raise HTTPException(status_code=422, detail="edge_mode must be 'duplicate' or 'none'")

    try:
        created = await svc.split_entity(
            tenant_id=tenant_id, doc_id=doc_id, config_path=CONFIG_PATH,
            entity_name=req.entity_name, entity_type=req.entity_type,
            new_entities=[e.model_dump() for e in req.new_entities],
            edge_mode=req.edge_mode,
            user_id=user_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        log.exception(f"split_entity failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return EntityOperationResponse(
        success=True,
        message=f"Split '{req.entity_name}' into {len(created)} entities",
        entities=[EntityInfo(**e) for e in created],
    )


# ── Suggest merges ────────────────────────────────────────────────────────────

@router.get("/{doc_id}/suggestions", response_model=SuggestMergesResponse)
async def suggest_merges(
    doc_id: str,
    min_score: float = 0.80,
    top_k: int = 50,
    use_embeddings: bool = False,
    current_user: dict = Depends(get_current_user),
):
    """Return ranked merge-candidate pairs based on string/embedding similarity."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    await _require_access(tenant_id, user_id, doc_id)

    if not (0.0 <= min_score <= 1.0):
        raise HTTPException(status_code=422, detail="min_score must be between 0.0 and 1.0")

    try:
        raw = await svc.suggest_merges(
            tenant_id=tenant_id, doc_id=doc_id, config_path=CONFIG_PATH,
            min_score=min_score, top_k=top_k, use_embeddings=use_embeddings,
        )
    except Exception as exc:
        log.exception(f"suggest_merges failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return SuggestMergesResponse(suggestions=[
        MergeSuggestion(
            entity_a=EntityRef(**s["entity_a"]),
            entity_b=EntityRef(**s["entity_b"]),
            score=s["score"],
            method=s["method"],
        )
        for s in raw
    ])

