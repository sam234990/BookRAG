"""Tenant management router (admin only)."""
import logging
from fastapi import APIRouter, Depends, HTTPException

from api.models.requests import TenantCreateRequest, TenantResponse, PermissionGrantRequest
from api.db import mongodb as db
from api.dependencies import (
    MONGO_URI, MONGO_DB_PREFIX, MONGO_SYSTEM_DB,
    get_current_user, require_admin,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/tenants", tags=["tenants"])


@router.post("", status_code=201)
async def create_tenant(req: TenantCreateRequest, _admin=Depends(require_admin)):
    """Create a new tenant (admin only)."""
    existing = await db.get_tenant(MONGO_URI, MONGO_SYSTEM_DB, req.tenant_id)
    if existing:
        raise HTTPException(status_code=409, detail="Tenant already exists")
    await db.create_tenant(MONGO_URI, MONGO_SYSTEM_DB, req.model_dump())
    return {"message": f"Tenant '{req.tenant_id}' created"}


@router.get("/{tenant_id}", response_model=TenantResponse)
async def get_tenant(tenant_id: str, current_user=Depends(get_current_user)):
    """Retrieve tenant info. Users can only see their own tenant."""
    if current_user["role"] != "admin" and current_user["tenant_id"] != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    tenant = await db.get_tenant(MONGO_URI, MONGO_SYSTEM_DB, tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return TenantResponse(
        tenant_id=tenant["tenant_id"],
        name=tenant.get("name", ""),
        description=tenant.get("description", ""),
    )


@router.post("/{tenant_id}/permissions", status_code=201)
async def grant_permission(
    tenant_id: str,
    req: PermissionGrantRequest,
    current_user=Depends(get_current_user),
):
    """Grant a user read access to a document within a tenant."""
    # Only admins or tenant members with admin role can grant permissions
    if current_user["role"] != "admin" and current_user["tenant_id"] != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    await db.grant_permission(
        MONGO_URI, MONGO_DB_PREFIX, tenant_id,
        req.user_id, req.doc_id, req.role,
    )
    return {"message": f"Permission granted: {req.user_id} → {req.doc_id} ({req.role})"}

