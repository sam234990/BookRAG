# BookRAG MCP Server — Implementation Guide

> **Model Context Protocol (MCP)** is an open standard by Anthropic that lets AI assistants (Claude Desktop, Cursor, Windsurf, etc.) connect directly to external tools and data sources through a unified interface. This document describes how to expose the BookRAG API as an MCP server so that AI agents can query books, inspect knowledge-graph entities, and manage documents — without any HTTP REST calls.

---

## Table of Contents

1. [What is MCP?](#1-what-is-mcp)
2. [Why BookRAG Maps Perfectly to MCP](#2-why-bookrag-maps-perfectly-to-mcp)
3. [Architecture](#3-architecture)
4. [MCP Primitives Mapping](#4-mcp-primitives-mapping)
5. [Multi-Tenancy Strategy](#5-multi-tenancy-strategy)
6. [Installation](#6-installation)
7. [File Structure](#7-file-structure)
8. [Implementation: `mcp_server.py`](#8-implementation-mcp_serverpy)
9. [Mounting to the Existing FastAPI App](#9-mounting-to-the-existing-fastapi-app)
10. [Claude Desktop & Cursor Configuration](#10-claude-desktop--cursor-configuration)
11. [Long-Running Operations (Indexing)](#11-long-running-operations-indexing)
12. [Testing](#12-testing)
13. [Transport Options](#13-transport-options)

---

## 1. What is MCP?

MCP defines three primitive types a server can expose:

| Primitive | Who controls it | Description | BookRAG example |
|---|---|---|---|
| **Resource** | Application | Read-only contextual data, URI-addressable | Entity list, document status |
| **Tool** | Model (LLM) | Callable functions that take actions | Query a book, rename an entity |
| **Prompt** | User | Reusable prompt templates | "Ask about book", "Find duplicates" |

MCP is **not** a replacement for REST. It is a parallel interface optimised for AI-agent consumption — same underlying service layer, different transport.

---

## 2. Why BookRAG Maps Perfectly to MCP

BookRAG's three-layer architecture is already MCP-ready:

```
Transport (HTTP REST)  →  api/routers/        ← keep as-is for users/web UI
Business logic         →  api/services/       ← shared, zero changes needed
Data stores            →  FalkorDB, MongoDB, ChromaDB
```

Adding MCP means writing a **thin new `mcp_server.py`** that calls the same `api/services/` functions — identical to how the FastAPI routers call them today.

Key reasons conversion is straightforward:
- All business logic is already **async Python** (`asyncio`)
- Services accept plain Python args — no HTTP concepts leak into them
- Per-document locking, thread pools, and FalkorDB persistence are all inside `api/services/`
- No changes needed to `Core/` indexing pipeline

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────┐
│               AI Agents / Claude Desktop            │
│         (Claude, Cursor, Windsurf, custom)          │
└────────────────────┬────────────────────────────────┘
                     │  MCP (stdio or streamable-http)
                     ▼
            ┌─────────────────┐
            │  mcp_server.py  │  ← NEW (thin adapter)
            └────────┬────────┘
                     │
        ┌────────────▼────────────┐
        │      api/services/      │  ← SHARED (unchanged)
        │  entity_editor.py       │
        │  chat.py                │
        │  indexing.py            │
        │  entity_resolution.py   │
        └──┬──────────┬──────────┘
           │          │
    ┌──────▼──┐  ┌────▼──────┐
    │FalkorDB │  │  MongoDB  │
    │ChromaDB │  │  uploads/ │
    └─────────┘  └───────────┘

┌─────────────────────────────────────────────────────┐
│            Web / Mobile Users                       │
└─────────────────┬───────────────────────────────────┘
                  │  HTTP REST (JSON)
                  ▼
         ┌─────────────────┐
         │  api/routers/   │  ← EXISTING FastAPI (unchanged)
         └─────────────────┘
```

Both interfaces use **the same service layer** — changes made through MCP are instantly visible through REST and vice versa.

---

## 4. MCP Primitives Mapping

### Resources (read-only data)

| URI pattern | Description | Backed by |
|---|---|---|
| `bookrag://documents/{tenant_id}` | List all documents for the tenant | `db.list_documents()` |
| `bookrag://documents/{tenant_id}/{doc_id}` | Single document status | `db.get_document()` |
| `bookrag://entities/{tenant_id}/{doc_id}` | All NER entities for a document | `entity_editor.list_entities()` |

Resources are **application-controlled**: the AI client decides when to read them as context, without the model explicitly calling a tool.

### Tools (callable by the model)

| Tool name | Maps to | Description |
|---|---|---|
| `query_documents` | `chat.handle_query()` | Ask a question against one or more indexed books |
| `rename_entity` | `entity_editor.rename_entity()` | Rename an entity node in the knowledge graph |
| `merge_entities` | `entity_editor.merge_entities()` | Merge ≥ 2 entity nodes into a canonical node |
| `split_entity` | `entity_editor.split_entity()` | Split 1 entity into ≥ 2 new nodes |
| `suggest_merge_candidates` | `entity_editor.suggest_merges()` | Find likely duplicate entities |
| `get_document_status` | `db.get_document()` | Check indexing status of a document |
| `index_document` | `indexing.run_indexing()` | Trigger indexing for an uploaded PDF |

### Prompts (user-invoked templates)

| Prompt name | Description |
|---|---|
| `ask_about_book` | Template: "Given document `{doc_id}`, answer: `{question}`" |
| `find_entity_duplicates` | Template: "Review these merge suggestions and decide which to apply" |
| `summarise_entities` | Template: "List the most important entities in `{doc_id}` and explain their roles" |

---

## 5. Multi-Tenancy Strategy

MCP has **no built-in JWT authentication**. Every service call needs a `tenant_id` and `user_id`. Three options:

### Option A — Environment-Variable Injection (Recommended)

Each tenant runs their own MCP server process. The `tenant_id` and `user_id` are injected via environment variables at launch time.

```
BOOKRAG_TENANT_ID=acme  BOOKRAG_USER_ID=alice  python mcp_server.py
```

Inside `mcp_server.py`:

```python
import os
TENANT_ID = os.environ["BOOKRAG_TENANT_ID"]   # required — fail fast if missing
USER_ID   = os.environ["BOOKRAG_USER_ID"]
```

**Pros**: Simple, no auth complexity, works with Claude Desktop `env` block.
**Cons**: One process per tenant — fine for small deployments.

### Option B — Tool-Argument Injection

`tenant_id` and `user_id` are required arguments on every tool. The AI model must supply them.

**Pros**: Single process for all tenants.
**Cons**: Verbose; model must always pass credentials; no real security boundary.

### Option C — OAuth 2.0 (MCP 1.1+)

MCP's newer spec supports OAuth 2.0 flows. Suitable for a SaaS product where the MCP server is hosted remotely and multiple organisations connect to it.

**Pros**: Proper per-user auth, scalable.
**Cons**: Requires implementing an OAuth server; significant complexity.

> **Recommendation for BookRAG**: Start with **Option A** (env-var injection). It matches exactly how Claude Desktop is configured and requires minimal code.

---

## 6. Installation

Add the MCP SDK to the project's virtual environment:

```bash
# Using pip (existing .venv)
pip install "mcp[cli]"

# Or with uv
uv add "mcp[cli]"
```

The `[cli]` extra installs the `mcp` command-line tool needed for the development inspector.

---

## 7. File Structure

Only **one new file** is needed at the repo root:

```
BookRAG/
├── mcp_server.py          ← NEW — MCP adapter (thin layer over api/services/)
├── api/
│   ├── main.py            ← existing FastAPI app (unchanged)
│   ├── services/          ← shared business logic (unchanged)
│   ├── routers/           ← existing REST endpoints (unchanged)
│   └── ...
├── Core/                  ← GBC indexing pipeline (unchanged)
└── config/
    └── gbc.yaml           ← existing config (unchanged)
```

Alternatively, for a remote/production deployment where MCP is mounted directly onto the FastAPI app, no new file is needed — see [Section 9](#9-mounting-to-the-existing-fastapi-app).

---

## 8. Implementation: `mcp_server.py`

Below is the complete implementation skeleton. It uses the **FastMCP** high-level API (`from mcp.server.fastmcp import FastMCP`) which is analogous to FastAPI's `APIRouter`.

```python
"""BookRAG MCP Server.

Usage (local / Claude Desktop):
    BOOKRAG_TENANT_ID=acme BOOKRAG_USER_ID=alice python mcp_server.py

Usage (dev inspector):
    BOOKRAG_TENANT_ID=acme BOOKRAG_USER_ID=alice mcp dev mcp_server.py
"""
import json
import os

from mcp.server.fastmcp import FastMCP, Context

# ── Tenant identity (injected via environment) ─────────────────────────────
TENANT_ID   = os.environ.get("BOOKRAG_TENANT_ID", "default")
USER_ID     = os.environ.get("BOOKRAG_USER_ID",   "agent")
CONFIG_PATH = os.environ.get("BOOKRAG_CONFIG_PATH", "config/gbc.yaml")

# ── Service imports (lazy, same as routers do) ──────────────────────────────
import api.services.entity_editor as entity_svc
import api.services.chat          as chat_svc
import api.services.indexing      as index_svc
import api.db.mongodb              as db

from api.dependencies import MONGO_URI, MONGO_DB_PREFIX

mcp = FastMCP("bookrag", instructions=(
    "BookRAG gives you access to a hierarchical RAG knowledge base built from PDF books. "
    "Use query_documents to ask questions. Use entity tools to inspect and curate the "
    "knowledge graph extracted from each book."
))


# ════════════════════════════════════════════════════════════════════════════
# RESOURCES  (read-only data — application-controlled)
# ════════════════════════════════════════════════════════════════════════════

@mcp.resource("bookrag://documents/{tenant_id}")
async def list_documents_resource(tenant_id: str) -> str:
    """Return the list of all documents for the given tenant as JSON."""
    docs = await db.list_documents(MONGO_URI, MONGO_DB_PREFIX, tenant_id, user_id=None)
    return json.dumps(docs, default=str)


@mcp.resource("bookrag://documents/{tenant_id}/{doc_id}")
async def get_document_resource(tenant_id: str, doc_id: str) -> str:
    """Return indexing status and metadata for a single document as JSON."""
    doc = await db.get_document(MONGO_URI, MONGO_DB_PREFIX, tenant_id, doc_id)
    return json.dumps(doc, default=str) if doc else json.dumps({"error": "not found"})


@mcp.resource("bookrag://entities/{tenant_id}/{doc_id}")
async def get_entities_resource(tenant_id: str, doc_id: str) -> str:
    """Return all NER entities for a document as a JSON array."""
    entities = await entity_svc.list_entities(tenant_id, doc_id, CONFIG_PATH)
    return json.dumps(entities, default=str)


# ════════════════════════════════════════════════════════════════════════════
# TOOLS  (callable by the model)
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def query_documents(
    question: str,
    doc_ids: list[str],
    session_id: str = "",
    cross_doc: bool = False,
) -> str:
    """Query one or more indexed books with a natural-language question.

    Args:
        question:   The question to ask.
        doc_ids:    List of document IDs to query (must be 'ready' status).
        session_id: Optional session ID to continue a conversation thread.
        cross_doc:  If True, query all docs in parallel and merge answers.
    """
    result = await chat_svc.handle_query(
        query=question,
        tenant_id=TENANT_ID,
        user_id=USER_ID,
        doc_ids=doc_ids,
        session_id=session_id or None,
        config_path=CONFIG_PATH,
        cross_doc=cross_doc,
    )
    return result.get("answer", str(result))


@mcp.tool()
async def get_document_status(doc_id: str) -> str:
    """Check the indexing status of a document (pending/indexing/ready/error)."""
    doc = await db.get_document(MONGO_URI, MONGO_DB_PREFIX, TENANT_ID, doc_id)
    if not doc:
        return f"Document '{doc_id}' not found."
    return json.dumps({
        "doc_id":   doc["doc_id"],
        "filename": doc.get("filename", ""),
        "status":   doc.get("status", "unknown"),
        "error":    doc.get("error"),
    })


@mcp.tool()
async def list_entities(doc_id: str) -> str:
    """Return all NER entities extracted from the knowledge graph of a document."""
    entities = await entity_svc.list_entities(TENANT_ID, doc_id, CONFIG_PATH)
    return json.dumps(entities, default=str)


@mcp.tool()
async def rename_entity(
    doc_id: str,
    entity_name: str,
    entity_type: str,
    new_entity_name: str,
    new_entity_type: str = "",
    new_description: str = "",
) -> str:
    """Rename an entity node in the knowledge graph.

    Args:
        doc_id:           Document the entity belongs to.
        entity_name:      Current entity name (exact match).
        entity_type:      Current entity type (e.g. PERSON, ORG).
        new_entity_name:  New name for the entity.
        new_entity_type:  New type (leave blank to keep current).
        new_description:  New description (leave blank to keep current).
    """
    updated = await entity_svc.rename_entity(
        tenant_id=TENANT_ID, doc_id=doc_id, config_path=CONFIG_PATH,
        entity_name=entity_name, entity_type=entity_type,
        new_entity_name=new_entity_name,
        new_entity_type=new_entity_type or entity_type,
        new_description=new_description or None,
        user_id=USER_ID,
    )
    return json.dumps(updated, default=str)


@mcp.tool()
async def merge_entities(
    doc_id: str,
    source_entities: list[dict],
    canonical_name: str,
    canonical_type: str,
    canonical_description: str = "",
) -> str:
    """Merge two or more entity nodes into a single canonical entity.

    Args:
        doc_id:                Document containing the entities.
        source_entities:       List of {"entity_name": ..., "entity_type": ...} dicts.
        canonical_name:        Name of the resulting merged entity.
        canonical_type:        Type of the resulting merged entity.
        canonical_description: Optional description for the canonical entity.
    """
    updated = await entity_svc.merge_entities(
        tenant_id=TENANT_ID, doc_id=doc_id, config_path=CONFIG_PATH,
        source_entities=source_entities,
        canonical_name=canonical_name,
        canonical_type=canonical_type,
        canonical_desc=canonical_description,
        user_id=USER_ID,
    )
    return json.dumps(updated, default=str)


@mcp.tool()
async def suggest_merge_candidates(
    doc_id: str,
    min_score: float = 0.80,
    top_k: int = 20,
    use_embeddings: bool = False,
) -> str:
    """Return a ranked list of entity pairs that may be duplicates.

    Args:
        doc_id:         Document to analyse.
        min_score:      Minimum similarity score (0.0 – 1.0). Default 0.80.
        top_k:          Maximum number of suggestions to return.
        use_embeddings: If True, also run embedding-based similarity (slower).
    """
    suggestions = await entity_svc.suggest_merges(
        tenant_id=TENANT_ID, doc_id=doc_id, config_path=CONFIG_PATH,
        min_score=min_score, top_k=top_k, use_embeddings=use_embeddings,
    )
    return json.dumps(suggestions, default=str)


@mcp.tool()
async def index_document(
    doc_id: str,
    pdf_path: str,
    ctx: Context,
) -> str:
    """Trigger GBC index build for an already-uploaded PDF.

    This is a long-running operation. Progress is reported via MCP notifications.

    Args:
        doc_id:   The document ID (must already exist in MongoDB).
        pdf_path: Absolute path to the PDF file on the server.
    """
    await ctx.report_progress(0, 100, "Starting indexing...")
    try:
        await index_svc.run_indexing(TENANT_ID, doc_id, pdf_path, CONFIG_PATH)
        await ctx.report_progress(100, 100, "Indexing complete.")
        return f"Document '{doc_id}' indexed successfully."
    except Exception as exc:
        return f"Indexing failed: {exc}"


# ════════════════════════════════════════════════════════════════════════════
# PROMPTS  (user-invoked templates)
# ════════════════════════════════════════════════════════════════════════════

@mcp.prompt()
def ask_about_book(doc_id: str, question: str) -> str:
    """Generate a prompt to ask a question about a specific indexed book."""
    return (
        f"You have access to the BookRAG knowledge base for document '{doc_id}'.\n\n"
        f"Please use the `query_documents` tool with doc_ids=['{doc_id}'] to answer:\n\n"
        f"{question}"
    )


@mcp.prompt()
def find_entity_duplicates(doc_id: str) -> str:
    """Generate a prompt to review and resolve duplicate entities."""
    return (
        f"You are reviewing the knowledge graph for document '{doc_id}'.\n\n"
        f"1. Call `suggest_merge_candidates(doc_id='{doc_id}', min_score=0.80)` to get candidates.\n"
        f"2. Review each pair. For genuine duplicates, call `merge_entities`.\n"
        f"3. Report a summary of what was merged and why."
    )


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Default: stdio transport for Claude Desktop / local use
    mcp.run(transport="stdio")
```

---

## 9. Mounting to the Existing FastAPI App

For **remote / production** deployments where you want one process serving both REST and MCP, mount the MCP server directly inside `api/main.py`:

```python
# api/main.py  (addition only — all existing code unchanged)
from mcp.server.fastmcp import FastMCP

# Import the mcp instance from your server module
from mcp_server import mcp   # the FastMCP instance defined above

# Mount under /mcp  — accessible at http://host:8000/mcp
app.mount("/mcp", mcp.streamable_http_app())
```

The MCP endpoint is then reachable at `http://your-server:8000/mcp` using the **streamable-http** transport. AI clients connect to this URL rather than launching a subprocess.

> **Note**: When mounted to FastAPI, the `mcp_server.py` `if __name__ == "__main__"` block is never executed. Uvicorn/Gunicorn drives everything.

---

## 10. Claude Desktop & Cursor Configuration

### Claude Desktop

Edit `~/.config/claude/claude_desktop_config.json` (Linux/macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "bookrag-acme": {
      "command": "/path/to/BookRAG/.venv/bin/python",
      "args": ["/path/to/BookRAG/mcp_server.py"],
      "env": {
        "BOOKRAG_TENANT_ID":    "acme",
        "BOOKRAG_USER_ID":      "alice",
        "BOOKRAG_CONFIG_PATH":  "/path/to/BookRAG/config/gbc.yaml",
        "BOOKRAG_UPLOAD_DIR":   "/path/to/BookRAG/uploads",
        "BOOKRAG_INDEX_DIR":    "/path/to/BookRAG/indices",
        "BOOKRAG_FALKORDB_HOST": "localhost",
        "BOOKRAG_FALKORDB_PORT": "6379",
        "MONGO_URI":            "mongodb://localhost:27017"
      }
    }
  }
}
```

Restart Claude Desktop. The BookRAG tools will appear in the 🔧 tools panel.

To support multiple tenants in Claude Desktop, add multiple entries with different `BOOKRAG_TENANT_ID` / `BOOKRAG_USER_ID` values:

```json
{
  "mcpServers": {
    "bookrag-acme":  { "command": "...", "env": { "BOOKRAG_TENANT_ID": "acme",  ... } },
    "bookrag-beta":  { "command": "...", "env": { "BOOKRAG_TENANT_ID": "beta",  ... } }
  }
}
```

### Cursor / Windsurf

In your project's `.cursor/mcp.json` (or Windsurf equivalent):

```json
{
  "mcpServers": {
    "bookrag": {
      "command": ".venv/bin/python",
      "args": ["mcp_server.py"],
      "env": {
        "BOOKRAG_TENANT_ID": "dev",
        "BOOKRAG_USER_ID":   "cursor-agent",
        "BOOKRAG_CONFIG_PATH": "config/gbc.yaml"
      }
    }
  }
}
```

### Remote (Streamable HTTP) Client Config

For clients that support the streamable-http transport (custom agents, LangChain, PydanticAI):

```python
from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession

async with streamable_http_client("http://your-server:8000/mcp") as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("query_documents", {
            "question": "Who is the main antagonist?",
            "doc_ids": ["doc-abc123"],
        })
        print(result.content[0].text)
```

---

## 11. Long-Running Operations (Indexing)

MCP tools are **request/response** — the client waits for the tool to return. Indexing a large PDF can take minutes. The `index_document` tool handles this with progress reporting:

```python
@mcp.tool()
async def index_document(doc_id: str, pdf_path: str, ctx: Context) -> str:
    await ctx.report_progress(0, 100, "Starting indexing...")
    await index_svc.run_indexing(TENANT_ID, doc_id, pdf_path, CONFIG_PATH)
    await ctx.report_progress(100, 100, "Done.")
    return f"Document '{doc_id}' indexed successfully."
```

`ctx.report_progress(current, total, message)` sends MCP progress notifications that Claude Desktop displays as a progress bar. The client tool call remains open until the function returns.

For very long operations (> 5 min), the recommended pattern is:
1. Launch indexing as a background task (already done in `run_indexing`)
2. Return immediately with `"Indexing started. Call get_document_status('{doc_id}') to check progress."`
3. The model can poll `get_document_status` in subsequent turns.

---

## 12. Testing

### Interactive MCP Inspector (recommended first step)

```bash
cd /path/to/BookRAG
BOOKRAG_TENANT_ID=dev BOOKRAG_USER_ID=test \
  mcp dev mcp_server.py
```

This opens a web UI at `http://localhost:5173` where you can:
- Browse all registered Resources, Tools, and Prompts
- Call any tool interactively and inspect the JSON response
- No Claude Desktop or Cursor needed

### Quick smoke-test with the MCP Python client

```python
# test_mcp_client.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(
        command=".venv/bin/python",
        args=["mcp_server.py"],
        env={
            "BOOKRAG_TENANT_ID": "test",
            "BOOKRAG_USER_ID":   "tester",
            "BOOKRAG_CONFIG_PATH": "config/gbc.yaml",
        },
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Tools:", [t.name for t in tools.tools])

            resources = await session.list_resources()
            print("Resources:", [r.uri for r in resources.resources])

asyncio.run(main())
```

Run with:

```bash
python test_mcp_client.py
```

---

## 13. Transport Options

| Transport | Use case | How to run |
|---|---|---|
| **stdio** | Local: Claude Desktop, Cursor, dev | `mcp.run(transport="stdio")` (default) |
| **streamable-http** | Remote: hosted server, custom agents | `app.mount("/mcp", mcp.streamable_http_app())` |
| **SSE** | Legacy remote (older MCP clients) | `mcp.run(transport="sse")` |

For most BookRAG deployments:
- **Development / single user**: stdio via Claude Desktop config
- **Team / production**: mount streamable-http on the existing FastAPI app at `/mcp`

---

## Summary

| Step | Action |
|---|---|
| 1 | `pip install "mcp[cli]"` |
| 2 | Create `mcp_server.py` (copy skeleton from Section 8) |
| 3 | Set env vars: `BOOKRAG_TENANT_ID`, `BOOKRAG_USER_ID`, `BOOKRAG_CONFIG_PATH` |
| 4 | Test with `mcp dev mcp_server.py` |
| 5 | Add to Claude Desktop config (Section 10) |
| 6 | *(Optional)* Mount to FastAPI for remote access (Section 9) |

**Zero changes** to `Core/`, `api/services/`, `api/routers/`, or any existing behaviour are required.

