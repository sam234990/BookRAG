"""Async MongoDB client and CRUD helpers using Motor."""
import hashlib
import logging
import os
from typing import List, Optional
from datetime import datetime, timezone

import pymongo
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

log = logging.getLogger(__name__)

_client: Optional[AsyncIOMotorClient] = None


def get_client(uri: str) -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(uri)
    return _client


def get_system_db(uri: str, system_db: str) -> AsyncIOMotorDatabase:
    return get_client(uri)[system_db]


def get_tenant_db(uri: str, db_prefix: str, tenant_id: str) -> AsyncIOMotorDatabase:
    return get_client(uri)[f"{db_prefix}_{tenant_id}"]


async def close_client():
    global _client
    if _client:
        _client.close()
        _client = None


async def ensure_indexes(uri: str, system_db: str, db_prefix: str, tenant_ids: List[str]):
    """Create MongoDB indexes for all known tenant databases.

    Called once at startup. Indexes are idempotent — ``create_index`` is a no-op
    if the index already exists.
    """
    client = get_client(uri)

    # System DB indexes
    sdb = client[system_db]
    await sdb["tenants"].create_index("tenant_id", unique=True)
    log.info(f"Ensured indexes on system db '{system_db}'")

    # Per-tenant DB indexes
    for tid in tenant_ids:
        tdb = client[f"{db_prefix}_{tid}"]
        await tdb["users"].create_index("username", unique=True)
        await tdb["users"].create_index("user_id", unique=True)
        await tdb["documents"].create_index("doc_id", unique=True)
        await tdb["permissions"].create_index(
            [("user_id", pymongo.ASCENDING), ("doc_id", pymongo.ASCENDING)],
            unique=True,
        )
        await tdb["sessions"].create_index("session_id", unique=True)
        await tdb["sessions"].create_index("user_id")
        await tdb["entity_edits"].create_index("doc_id")
        await tdb["entity_edits"].create_index("ts")
        # Refresh token revocation index
        await tdb["refresh_tokens"].create_index("token_hash", unique=True)
        await tdb["refresh_tokens"].create_index("user_id")
        await tdb["refresh_tokens"].create_index("expires_at", expireAfterSeconds=0)
    log.info(f"Ensured indexes on {len(tenant_ids)} tenant database(s)")


# ── Tenant CRUD ──────────────────────────────────────────────────────────────

async def create_tenant(uri: str, system_db: str, tenant_data: dict) -> str:
    db = get_system_db(uri, system_db)
    tenant_data["created_at"] = datetime.now(timezone.utc)
    result = await db["tenants"].insert_one(tenant_data)
    return str(result.inserted_id)


async def get_tenant(uri: str, system_db: str, tenant_id: str) -> Optional[dict]:
    db = get_system_db(uri, system_db)
    return await db["tenants"].find_one({"tenant_id": tenant_id})


# ── User CRUD ─────────────────────────────────────────────────────────────────

async def create_user(uri: str, db_prefix: str, tenant_id: str, user_data: dict) -> str:
    db = get_tenant_db(uri, db_prefix, tenant_id)
    user_data["created_at"] = datetime.now(timezone.utc)
    result = await db["users"].insert_one(user_data)
    return str(result.inserted_id)


async def get_user_by_username(uri: str, db_prefix: str, tenant_id: str, username: str) -> Optional[dict]:
    db = get_tenant_db(uri, db_prefix, tenant_id)
    return await db["users"].find_one({"username": username})


# ── Document CRUD ─────────────────────────────────────────────────────────────

async def create_document(uri: str, db_prefix: str, tenant_id: str, doc_data: dict) -> str:
    db = get_tenant_db(uri, db_prefix, tenant_id)
    doc_data["created_at"] = datetime.now(timezone.utc)
    doc_data["status"] = "pending"
    result = await db["documents"].insert_one(doc_data)
    return str(result.inserted_id)


async def update_document_status(uri: str, db_prefix: str, tenant_id: str, doc_id: str, status: str, error: str = None):
    db = get_tenant_db(uri, db_prefix, tenant_id)
    update = {"$set": {"status": status, "updated_at": datetime.now(timezone.utc)}}
    if error:
        update["$set"]["error"] = error
    await db["documents"].update_one({"doc_id": doc_id}, update)


async def get_document(uri: str, db_prefix: str, tenant_id: str, doc_id: str) -> Optional[dict]:
    db = get_tenant_db(uri, db_prefix, tenant_id)
    return await db["documents"].find_one({"doc_id": doc_id})


async def get_document_raw_path(uri: str, db_prefix: str, tenant_id: str, doc_id: str) -> Optional[str]:
    """Return the raw PDF path stored at upload time, or None if not found."""
    doc = await get_document(uri, db_prefix, tenant_id, doc_id)
    return doc.get("pdf_path") if doc else None


async def list_documents(
    uri: str, db_prefix: str, tenant_id: str, user_id: str,
    limit: int = 50, offset: int = 0,
) -> tuple[list[dict], int]:
    """Return paginated docs accessible to *user_id*, sorted by document_date desc.

    Returns ``(docs, total_count)``.  Documents without ``document_date``
    fall back to ``created_at``; documents with neither sort last.
    """
    db = get_tenant_db(uri, db_prefix, tenant_id)
    perm_cursor = db["permissions"].find({"user_id": user_id})
    doc_ids = [p["doc_id"] async for p in perm_cursor]
    filt = {"doc_id": {"$in": doc_ids}}
    total = await db["documents"].count_documents(filt)
    cursor = (
        db["documents"]
        .find(filt)
        .sort([("document_date", pymongo.DESCENDING), ("created_at", pymongo.DESCENDING)])
        .skip(offset)
        .limit(limit)
    )
    docs = [d async for d in cursor]
    return docs, total


async def delete_document(uri: str, db_prefix: str, tenant_id: str, doc_id: str):
    """Delete a document and all associated permissions, sessions, entity edits."""
    db = get_tenant_db(uri, db_prefix, tenant_id)
    await db["documents"].delete_one({"doc_id": doc_id})
    await db["permissions"].delete_many({"doc_id": doc_id})
    await db["entity_edits"].delete_many({"doc_id": doc_id})
    log.info(f"Deleted document '{doc_id}' and related records from tenant '{tenant_id}'")


# ── Permission CRUD ───────────────────────────────────────────────────────────

async def grant_permission(uri: str, db_prefix: str, tenant_id: str, user_id: str, doc_id: str, role: str = "reader"):
    db = get_tenant_db(uri, db_prefix, tenant_id)
    await db["permissions"].update_one(
        {"user_id": user_id, "doc_id": doc_id},
        {"$set": {"role": role, "updated_at": datetime.now(timezone.utc)}},
        upsert=True,
    )


async def get_accessible_doc_ids(uri: str, db_prefix: str, tenant_id: str, user_id: str) -> List[str]:
    db = get_tenant_db(uri, db_prefix, tenant_id)
    cursor = db["permissions"].find({"user_id": user_id})
    return [p["doc_id"] async for p in cursor]


async def get_permission(uri: str, db_prefix: str, tenant_id: str, user_id: str, doc_id: str) -> Optional[dict]:
    """Return the permission record for a user+doc pair, or None."""
    db = get_tenant_db(uri, db_prefix, tenant_id)
    return await db["permissions"].find_one({"user_id": user_id, "doc_id": doc_id})


# ── Session / Message CRUD ────────────────────────────────────────────────────

async def create_session(uri: str, db_prefix: str, tenant_id: str, session_data: dict) -> str:
    db = get_tenant_db(uri, db_prefix, tenant_id)
    session_data["created_at"] = datetime.now(timezone.utc)
    result = await db["sessions"].insert_one(session_data)
    return str(result.inserted_id)


# Max messages per session — prevents unbounded array growth (16 MB doc limit).
# Each user+assistant turn = 2 messages, so 200 = ~100 turns.
_SESSION_MSG_CAP = int(os.environ.get("BOOKRAG_SESSION_MSG_CAP", "200"))


async def append_message(uri: str, db_prefix: str, tenant_id: str, session_id: str, message: dict):
    db = get_tenant_db(uri, db_prefix, tenant_id)
    message["ts"] = datetime.now(timezone.utc)
    await db["sessions"].update_one(
        {"session_id": session_id},
        {"$push": {"messages": {"$each": [message], "$slice": -_SESSION_MSG_CAP}}},
    )


async def get_session(uri: str, db_prefix: str, tenant_id: str, session_id: str) -> Optional[dict]:
    db = get_tenant_db(uri, db_prefix, tenant_id)
    return await db["sessions"].find_one({"session_id": session_id})


async def list_sessions(
    uri: str, db_prefix: str, tenant_id: str, user_id: str,
    limit: int = 50, offset: int = 0,
) -> tuple[list[dict], int]:
    """Return paginated sessions for *user_id*, newest first."""
    db = get_tenant_db(uri, db_prefix, tenant_id)
    filt = {"user_id": user_id}
    total = await db["sessions"].count_documents(filt)
    cursor = (
        db["sessions"]
        .find(filt, {"messages": 0})  # exclude messages array for listing
        .sort("created_at", pymongo.DESCENDING)
        .skip(offset)
        .limit(limit)
    )
    sessions = [s async for s in cursor]
    return sessions, total


async def delete_session(uri: str, db_prefix: str, tenant_id: str, session_id: str):
    """Delete a session and all its messages."""
    db = get_tenant_db(uri, db_prefix, tenant_id)
    await db["sessions"].delete_one({"session_id": session_id})


async def recover_stale_indexing(uri: str, db_prefix: str, tenant_ids: List[str]) -> int:
    """Reset docs stuck in 'indexing' status to 'error' — called at startup.

    If the server crashed mid-indexing, those docs will never complete.
    Returns the total number of documents recovered across all tenants.
    """
    client = get_client(uri)
    recovered = 0
    for tid in tenant_ids:
        tdb = client[f"{db_prefix}_{tid}"]
        result = await tdb["documents"].update_many(
            {"status": "indexing"},
            {"$set": {
                "status": "error",
                "error": "Indexing interrupted by server restart — please re-upload",
                "updated_at": datetime.now(timezone.utc),
            }},
        )
        recovered += result.modified_count
    return recovered


# ── Entity Edit Audit Log ─────────────────────────────────────────────────────

async def log_entity_edit(uri: str, db_prefix: str, tenant_id: str, edit_record: dict):
    """Write a lightweight audit entry for any entity edit operation.

    ``edit_record`` should contain at minimum:
        - ``operation``: one of "rename" | "merge" | "split"
        - ``doc_id``: document scope of the edit
        - ``user_id``: who performed the edit
        - ``before``: snapshot of the entity/entities before the change
        - ``after``:  snapshot of the entity/entities after the change
    """
    db = get_tenant_db(uri, db_prefix, tenant_id)
    edit_record["ts"] = datetime.now(timezone.utc)
    await db["entity_edits"].insert_one(edit_record)



# ── Refresh Token Management ────────────────────────────────────────────────

def _hash_token(token: str) -> str:
    """SHA-256 hash of the raw refresh token — we never store plaintext."""
    return hashlib.sha256(token.encode()).hexdigest()


async def store_refresh_token(
    uri: str, db_prefix: str, tenant_id: str,
    user_id: str, token: str, expires_at: datetime,
):
    """Store a hashed refresh token so it can be revoked later."""
    db = get_tenant_db(uri, db_prefix, tenant_id)
    await db["refresh_tokens"].insert_one({
        "token_hash": _hash_token(token),
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc),
        "expires_at": expires_at,
    })


async def is_refresh_token_valid(uri: str, db_prefix: str, tenant_id: str, token: str) -> bool:
    """Return True if the token hash exists (i.e. has NOT been revoked)."""
    db = get_tenant_db(uri, db_prefix, tenant_id)
    doc = await db["refresh_tokens"].find_one({"token_hash": _hash_token(token)})
    return doc is not None


async def revoke_refresh_token(uri: str, db_prefix: str, tenant_id: str, token: str):
    """Revoke a single refresh token by removing it."""
    db = get_tenant_db(uri, db_prefix, tenant_id)
    await db["refresh_tokens"].delete_one({"token_hash": _hash_token(token)})


async def revoke_all_refresh_tokens(uri: str, db_prefix: str, tenant_id: str, user_id: str):
    """Revoke all refresh tokens for a user (e.g. password change)."""
    db = get_tenant_db(uri, db_prefix, tenant_id)
    result = await db["refresh_tokens"].delete_many({"user_id": user_id})
    log.info(f"Revoked {result.deleted_count} refresh tokens for user {user_id}")
