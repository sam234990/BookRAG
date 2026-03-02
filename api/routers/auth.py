"""Authentication router: register and login endpoints."""
import logging
from fastapi import APIRouter, HTTPException, status

from api.models.requests import RegisterRequest, LoginRequest, TokenResponse
from api.db import mongodb as db
from api.dependencies import (
    MONGO_URI, MONGO_DB_PREFIX, MONGO_SYSTEM_DB,
    hash_password, verify_password, create_access_token,
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


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """Authenticate and return a JWT access token."""
    user = await db.get_user_by_username(MONGO_URI, MONGO_DB_PREFIX, req.tenant_id, req.username)
    if not user or not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({
        "sub": user["user_id"],
        "tenant_id": user["tenant_id"],
        "role": user.get("role", "user"),
    })
    return TokenResponse(access_token=token)

