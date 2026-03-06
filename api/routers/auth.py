"""Authentication router: register, login, refresh, and logout endpoints."""
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from api.models.requests import RegisterRequest, LoginRequest, TokenResponse, RefreshRequest
from api.db import mongodb as db
from api.dependencies import (
    MONGO_URI, MONGO_DB_PREFIX, MONGO_SYSTEM_DB,
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_refresh_token,
    REFRESH_TOKEN_EXPIRE_DAYS,
    rate_limit_login, get_current_user,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest):
    """Register a new user within a tenant."""
    # Verify tenant exists
    tenant = await db.get_tenant(MONGO_URI, MONGO_SYSTEM_DB, req.tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail=f"Tenant '{req.tenant_id}' not found")

    # Check username not taken
    existing = await db.get_user_by_username(MONGO_URI, MONGO_DB_PREFIX, req.tenant_id, req.username)
    if existing:
        raise HTTPException(status_code=409, detail="Username already registered")

    user_data = {
        "username": req.username,
        "hashed_password": hash_password(req.password),
        "tenant_id": req.tenant_id,
        "role": "user",
        "user_id": req.username,  # use username as user_id for simplicity
    }
    await db.create_user(MONGO_URI, MONGO_DB_PREFIX, req.tenant_id, user_data)
    return {"message": "User registered successfully"}


@router.post("/login", response_model=TokenResponse, dependencies=[Depends(rate_limit_login)])
async def login(req: LoginRequest):
    """Authenticate and return JWT access + refresh tokens."""
    user = await db.get_user_by_username(MONGO_URI, MONGO_DB_PREFIX, req.tenant_id, req.username)
    if not user or not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    claims = {
        "sub": user["user_id"],
        "tenant_id": user["tenant_id"],
        "role": user.get("role", "user"),
    }
    access = create_access_token(claims)
    refresh = create_refresh_token(claims)

    # Store refresh token hash for revocation support
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    await db.store_refresh_token(
        MONGO_URI, MONGO_DB_PREFIX, req.tenant_id,
        user["user_id"], refresh, expires_at,
    )
    return TokenResponse(access_token=access, refresh_token=refresh)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(req: RefreshRequest):
    """Exchange a valid refresh token for a new access + refresh token pair.

    The old refresh token is revoked (single-use rotation).
    """
    payload = decode_refresh_token(req.refresh_token)
    tenant_id = payload.get("tenant_id", "")
    user_id = payload.get("sub", "")

    # Check the token hasn't been revoked
    if not await db.is_refresh_token_valid(MONGO_URI, MONGO_DB_PREFIX, tenant_id, req.refresh_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked",
        )

    # Revoke old refresh token (single-use rotation)
    await db.revoke_refresh_token(MONGO_URI, MONGO_DB_PREFIX, tenant_id, req.refresh_token)

    # Issue new pair
    claims = {"sub": user_id, "tenant_id": tenant_id, "role": payload.get("role", "user")}
    new_access = create_access_token(claims)
    new_refresh = create_refresh_token(claims)

    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    await db.store_refresh_token(MONGO_URI, MONGO_DB_PREFIX, tenant_id, user_id, new_refresh, expires_at)

    return TokenResponse(access_token=new_access, refresh_token=new_refresh)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(req: RefreshRequest, current_user: dict = Depends(get_current_user)):
    """Revoke the provided refresh token (logout)."""
    tenant_id = current_user["tenant_id"]
    await db.revoke_refresh_token(MONGO_URI, MONGO_DB_PREFIX, tenant_id, req.refresh_token)

