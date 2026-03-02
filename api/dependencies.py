"""FastAPI dependency injection: JWT verification, DB handles, permission checks."""
import os
import logging
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from api.db import mongodb as db

log = logging.getLogger(__name__)

# ── Config (read from env with sensible defaults) ─────────────────────────────
SECRET_KEY = os.getenv("BOOKRAG_SECRET_KEY", "change-me-in-production-please")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("BOOKRAG_TOKEN_EXPIRE", "60"))

MONGO_URI = os.getenv("BOOKRAG_MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_PREFIX = os.getenv("BOOKRAG_MONGO_PREFIX", "bookrag")
MONGO_SYSTEM_DB = os.getenv("BOOKRAG_MONGO_SYSTEM_DB", "bookrag_system")

FALKORDB_HOST = os.getenv("BOOKRAG_FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("BOOKRAG_FALKORDB_PORT", "6379"))
FALKORDB_PASSWORD = os.getenv("BOOKRAG_FALKORDB_PASSWORD", "")

UPLOAD_DIR = os.getenv("BOOKRAG_UPLOAD_DIR", "./uploads")
INDEX_SAVE_DIR = os.getenv("BOOKRAG_INDEX_DIR", "./indices")

# ── Auth helpers ──────────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict) -> str:
    from datetime import datetime, timedelta, timezone
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


# ── Current-user dependency ───────────────────────────────────────────────────

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        tenant_id: str = payload.get("tenant_id")
        role: str = payload.get("role", "user")
        if not user_id or not tenant_id:
            raise credentials_exc
    except JWTError:
        raise credentials_exc
    return {"user_id": user_id, "tenant_id": tenant_id, "role": role}


async def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user


# ── Permission check ─────────────────────────────────────────────────────────

async def check_doc_access(user_id: str, tenant_id: str, doc_id: str) -> bool:
    accessible = await db.get_accessible_doc_ids(MONGO_URI, MONGO_DB_PREFIX, tenant_id, user_id)
    return doc_id in accessible


async def filter_accessible_docs(user_id: str, tenant_id: str, requested_doc_ids: Optional[list]) -> list:
    """Return intersection of requested_doc_ids with what user can access. If requested is None, return all accessible."""
    accessible = await db.get_accessible_doc_ids(MONGO_URI, MONGO_DB_PREFIX, tenant_id, user_id)
    if requested_doc_ids is None:
        return accessible
    return [d for d in requested_doc_ids if d in accessible]

