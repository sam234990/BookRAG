"""Async MongoDB client and CRUD helpers using Motor."""
import logging
from typing import List, Optional
from datetime import datetime, timezone

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


async def list_documents(uri: str, db_prefix: str, tenant_id: str, user_id: str) -> List[dict]:
    db = get_tenant_db(uri, db_prefix, tenant_id)
    # Return docs the user has access to via permissions
    perm_cursor = db["permissions"].find({"user_id": user_id})
    doc_ids = [p["doc_id"] async for p in perm_cursor]
    cursor = db["documents"].find({"doc_id": {"$in": doc_ids}})
    return [d async for d in cursor]


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


# ── Session / Message CRUD ────────────────────────────────────────────────────

async def create_session(uri: str, db_prefix: str, tenant_id: str, session_data: dict) -> str:
    db = get_tenant_db(uri, db_prefix, tenant_id)
    session_data["created_at"] = datetime.now(timezone.utc)
    result = await db["sessions"].insert_one(session_data)
    return str(result.inserted_id)


async def append_message(uri: str, db_prefix: str, tenant_id: str, session_id: str, message: dict):
    db = get_tenant_db(uri, db_prefix, tenant_id)
    message["ts"] = datetime.now(timezone.utc)
    await db["sessions"].update_one(
        {"session_id": session_id},
        {"$push": {"messages": message}},
    )


async def get_session(uri: str, db_prefix: str, tenant_id: str, session_id: str) -> Optional[dict]:
    db = get_tenant_db(uri, db_prefix, tenant_id)
    return await db["sessions"].find_one({"session_id": session_id})


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

