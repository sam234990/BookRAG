"""Entity editor service: rename, merge, split, suggest-merges on NER entities.

All mutating operations work on the in-memory NetworkX graph (loaded from
graph_data.json), persist changes to graph_data.json *and* FalkorDB (when
configured), then best-effort rebuild the entity VDB so search stays fresh.

Entities are NOT stored in MongoDB — their source of truth is FalkorDB +
graph_data.json.  A lightweight audit entry is written to MongoDB's
``entity_edits`` collection for every mutating operation.
"""
from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from api.dependencies import (
    FALKORDB_HOST, FALKORDB_PORT, FALKORDB_PASSWORD,
    INDEX_SAVE_DIR, MONGO_URI, MONGO_DB_PREFIX,
    THREAD_POOL,
)
from api.db import mongodb as db

log = logging.getLogger(__name__)
_executor = THREAD_POOL

# Per-document asyncio lock — keyed by "{tenant_id}:{doc_id}"
_doc_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

# Dedup tracker: prevents redundant VDB rebuilds when multiple edits fire
# in quick succession on the same document.
_rebuild_pending: set[str] = set()
_rebuild_pending_lock = asyncio.Lock()


def _get_lock(tenant_id: str, doc_id: str) -> asyncio.Lock:
    return _doc_locks[f"{tenant_id}:{doc_id}"]


# ── Graph loader ─────────────────────────────────────────────────────────────

def _load_graph_sync(tenant_id: str, doc_id: str, config_path: str):
    """Load a Graph from JSON (never from FalkorDB) for in-memory editing.

    Returns ``(graph, save_path, falkordb_cfg | None)``.
    """
    from Core.configs.system_config import load_system_config
    from Core.configs.falkordb_config import FalkorDBConfig
    from Core.Index.Graph import Graph

    cfg = load_system_config(config_path)
    save_path = os.path.join(INDEX_SAVE_DIR, tenant_id, doc_id)
    variant = "basic" if cfg.graph.refine_type == "basic" else None

    # Always load from JSON so we get the full in-memory graph
    graph = Graph.load_from_dir(
        load_dir=save_path,
        variant=variant,
        tenant_id=tenant_id,
        doc_id=doc_id,
        falkordb_cfg=None,  # load from JSON only
    )

    # Attach FalkorDB cfg for saving if the host env var is set
    falkordb_cfg = None
    fdb_host = os.getenv("BOOKRAG_FALKORDB_HOST", "")
    if fdb_host:
        falkordb_cfg = FalkorDBConfig(
            host=FALKORDB_HOST,
            port=FALKORDB_PORT,
            password=FALKORDB_PASSWORD,
        )
        graph.falkordb_cfg = falkordb_cfg
        graph.tenant_id = tenant_id
        graph.doc_id = doc_id
        graph.use_falkordb = True
        graph._fdb_graph_name = falkordb_cfg.graph_name_for_doc(tenant_id, doc_id)

    return graph, save_path, falkordb_cfg


def _rebuild_vdb_sync(tenant_id: str, doc_id: str, config_path: str) -> None:
    """Best-effort VDB rebuild after any graph mutation."""
    try:
        from Core.configs.system_config import load_system_config
        from Core.configs.falkordb_config import FalkorDBConfig
        from Core.Index.GBCIndex import GBC

        cfg = load_system_config(config_path)
        cfg.tenant_id = tenant_id
        cfg.doc_id = doc_id
        cfg.save_path = os.path.join(INDEX_SAVE_DIR, tenant_id, doc_id)

        fdb_host = os.getenv("BOOKRAG_FALKORDB_HOST", "")
        if fdb_host:
            cfg.falkordb = FalkorDBConfig(
                host=FALKORDB_HOST, port=FALKORDB_PORT, password=FALKORDB_PASSWORD
            )

        gbc = GBC.load_gbc_index(cfg)
        gbc.rebuild_vdb()
        log.info(f"VDB rebuilt for {tenant_id}/{doc_id}")
    except Exception as exc:
        log.warning(f"VDB rebuild failed for {tenant_id}/{doc_id}: {exc}")


async def _schedule_vdb_rebuild(tenant_id: str, doc_id: str, config_path: str) -> None:
    """Await a VDB rebuild, deduplicating concurrent requests for the same doc.

    If a rebuild is already in-flight for this ``tenant_id:doc_id``, the call
    is skipped (the already-running rebuild will pick up the latest graph JSON).
    """
    key = f"{tenant_id}:{doc_id}"
    async with _rebuild_pending_lock:
        if key in _rebuild_pending:
            log.debug(f"VDB rebuild already pending for {key} — skipping")
            return
        _rebuild_pending.add(key)
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, _rebuild_vdb_sync, tenant_id, doc_id, config_path)
    finally:
        async with _rebuild_pending_lock:
            _rebuild_pending.discard(key)


# ── List entities ─────────────────────────────────────────────────────────────

def _list_entities_sync(tenant_id: str, doc_id: str, config_path: str) -> List[dict]:
    graph, _, _ = _load_graph_sync(tenant_id, doc_id, config_path)
    result = []
    for node_name in graph.get_all_nodes():
        entity = graph.get_entity_by_node_name(node_name)
        result.append({
            "entity_name": entity.entity_name,
            "entity_type": entity.entity_type,
            "description": entity.description,
            "source_ids": sorted(entity.source_ids),
            "node_name": node_name,
        })
    return sorted(result, key=lambda e: e["entity_name"].lower())


async def list_entities(tenant_id: str, doc_id: str, config_path: str) -> List[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, _list_entities_sync, tenant_id, doc_id, config_path
    )


# ── Rename entity ─────────────────────────────────────────────────────────────

def _rename_sync(
    tenant_id: str, doc_id: str, config_path: str,
    entity_name: str, entity_type: str,
    new_entity_name: str, new_entity_type: str, new_description: Optional[str],
) -> List[dict]:
    from Core.Index.Graph import Entity

    graph, _, _ = _load_graph_sync(tenant_id, doc_id, config_path)
    old_entity = graph.get_entity(entity_name, entity_type)

    effective_type = new_entity_type if new_entity_type else old_entity.entity_type
    effective_desc = new_description if new_description is not None else old_entity.description

    new_entity = Entity(
        entity_name=new_entity_name,
        entity_type=effective_type,
        description=effective_desc,
        source_ids=old_entity.source_ids,
    )
    graph.update_entity(entity_name, entity_type, new_entity)
    graph.save_graph()

    new_node = graph.get_node_name_from_str(new_entity_name, effective_type)
    return [{
        "entity_name": new_entity_name,
        "entity_type": effective_type,
        "description": effective_desc,
        "source_ids": sorted(new_entity.source_ids),
        "node_name": new_node,
    }]


async def rename_entity(
    tenant_id: str, doc_id: str, config_path: str,
    entity_name: str, entity_type: str,
    new_entity_name: str, new_entity_type: str, new_description: Optional[str],
    user_id: str,
) -> List[dict]:
    async with _get_lock(tenant_id, doc_id):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, _rename_sync,
            tenant_id, doc_id, config_path,
            entity_name, entity_type, new_entity_name, new_entity_type, new_description,
        )
    await db.log_entity_edit(MONGO_URI, MONGO_DB_PREFIX, tenant_id, {
        "operation": "rename", "doc_id": doc_id, "user_id": user_id,
        "before": {"entity_name": entity_name, "entity_type": entity_type},
        "after": {"entity_name": new_entity_name, "entity_type": new_entity_type or entity_type},
    })
    await _schedule_vdb_rebuild(tenant_id, doc_id, config_path)
    return result


# ── Split entity ──────────────────────────────────────────────────────────────

def _split_sync(
    tenant_id: str, doc_id: str, config_path: str,
    entity_name: str, entity_type: str,
    new_entities: List[dict],
    edge_mode: str,
) -> List[dict]:
    from Core.Index.Graph import Entity

    graph, _, _ = _load_graph_sync(tenant_id, doc_id, config_path)
    old_node = graph.get_node_name_from_str(entity_name, entity_type)
    if old_node not in graph.kg:
        raise KeyError(f"Entity '{old_node}' not found in graph.")

    old_entity = graph.get_entity_by_node_name(old_node)
    old_neighbors = list(graph.kg.neighbors(old_node))
    old_edge_data = {n: graph.kg.get_edge_data(old_node, n) for n in old_neighbors}

    created: List[dict] = []
    for spec in new_entities:
        spec_name = spec["entity_name"]
        spec_type = spec["entity_type"]
        spec_desc = spec.get("description") or old_entity.description
        spec_sids = set(spec.get("source_ids") or old_entity.source_ids)

        new_node = graph.get_node_name_from_str(spec_name, spec_type)
        graph.add_kg_node(Entity(
            entity_name=spec_name, entity_type=spec_type,
            description=spec_desc, source_ids=spec_sids,
        ))

        if edge_mode == "duplicate":
            for neighbor, edata in old_edge_data.items():
                if neighbor != old_node and not graph.kg.has_edge(new_node, neighbor):
                    graph.kg.add_edge(new_node, neighbor, **edata)

        for tree_id in spec_sids:
            graph.tree2kg[tree_id].add(new_node)

        created.append({
            "entity_name": spec_name, "entity_type": spec_type,
            "description": spec_desc, "source_ids": sorted(spec_sids), "node_name": new_node,
        })

    for _, nodes in graph.tree2kg.items():
        nodes.discard(old_node)
    graph.kg.remove_node(old_node)
    graph.save_graph()
    return created


async def split_entity(
    tenant_id: str, doc_id: str, config_path: str,
    entity_name: str, entity_type: str,
    new_entities: List[dict], edge_mode: str,
    user_id: str,
) -> List[dict]:
    async with _get_lock(tenant_id, doc_id):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, _split_sync,
            tenant_id, doc_id, config_path,
            entity_name, entity_type, new_entities, edge_mode,
        )
    await db.log_entity_edit(MONGO_URI, MONGO_DB_PREFIX, tenant_id, {
        "operation": "split", "doc_id": doc_id, "user_id": user_id,
        "before": {"entity_name": entity_name, "entity_type": entity_type},
        "after": [{"entity_name": e["entity_name"], "entity_type": e["entity_type"]} for e in result],
    })
    await _schedule_vdb_rebuild(tenant_id, doc_id, config_path)
    return result


# ── Suggest merge candidates ──────────────────────────────────────────────────

def _suggest_merges_sync(
    tenant_id: str, doc_id: str, config_path: str,
    min_score: float, top_k: int, use_embeddings: bool,
) -> List[dict]:
    graph, _, _ = _load_graph_sync(tenant_id, doc_id, config_path)
    nodes = list(graph.get_all_nodes())
    entities = []
    for node in nodes:
        ent = graph.get_entity_by_node_name(node)
        entities.append({"entity_name": ent.entity_name, "entity_type": ent.entity_type, "node": node})

    suggestions: List[dict] = []

    # Pre-group entities by type so we only compare within same-type groups
    # This reduces O(n²) to O(Σ nᵢ²) where nᵢ is entity count per type
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for ent in entities:
        by_type[ent["entity_type"]].append(ent)

    # String similarity — only within same-type groups
    for _etype, group in by_type.items():
        n = len(group)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = group[i], group[j]
                score = SequenceMatcher(None, a["entity_name"].lower(), b["entity_name"].lower()).ratio()
                if score >= min_score:
                    suggestions.append({
                        "entity_a": {"entity_name": a["entity_name"], "entity_type": a["entity_type"]},
                        "entity_b": {"entity_name": b["entity_name"], "entity_type": b["entity_type"]},
                        "score": round(score, 4),
                        "method": "string_similarity",
                    })

    # Embedding similarity (optional)
    if use_embeddings:
        try:
            from Core.configs.system_config import load_system_config
            from Core.Index.GBCIndex import GBC

            cfg = load_system_config(config_path)
            cfg.tenant_id = tenant_id
            cfg.doc_id = doc_id
            cfg.save_path = os.path.join(INDEX_SAVE_DIR, tenant_id, doc_id)
            gbc = GBC.load_gbc_index(cfg)

            seen_pairs: set = set()
            for ent in entities:
                for hit in gbc.entity_vdb.search(ent["node"], top_k=5):
                    sim = 1.0 - hit["distance"]
                    if sim < min_score:
                        continue
                    meta = hit.get("metadata", {})
                    b_name = meta.get("entity_name", "")
                    b_type = meta.get("entity_type", "")
                    if (b_name == ent["entity_name"] and b_type == ent["entity_type"]) or b_type != ent["entity_type"]:
                        continue
                    pair = tuple(sorted([(ent["entity_name"], ent["entity_type"]), (b_name, b_type)]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    suggestions.append({
                        "entity_a": {"entity_name": ent["entity_name"], "entity_type": ent["entity_type"]},
                        "entity_b": {"entity_name": b_name, "entity_type": b_type},
                        "score": round(sim, 4),
                        "method": "embedding_similarity",
                    })
        except Exception as exc:
            log.warning(f"Embedding suggestions failed: {exc}")

    suggestions.sort(key=lambda s: s["score"], reverse=True)
    return suggestions[:top_k]


async def suggest_merges(
    tenant_id: str, doc_id: str, config_path: str,
    min_score: float = 0.80,
    top_k: int = 50,
    use_embeddings: bool = False,
) -> List[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, _suggest_merges_sync,
        tenant_id, doc_id, config_path, min_score, top_k, use_embeddings,
    )
# ── Merge entities ────────────────────────────────────────────────────────────

def _merge_sync(
    tenant_id: str, doc_id: str, config_path: str,
    source_entities: List[dict],
    canonical_name: str, canonical_type: str, canonical_desc: str,
) -> List[dict]:
    from Core.Index.Graph import Entity

    graph, _, _ = _load_graph_sync(tenant_id, doc_id, config_path)

    # Collect all source_ids from entities being merged
    merged_source_ids: set = set()
    for src in source_entities:
        try:
            ent = graph.get_entity(src["entity_name"], src["entity_type"])
            merged_source_ids.update(ent.source_ids)
        except KeyError:
            log.warning(f"Merge: source entity not found: {src}")

    canonical_node = graph.get_node_name_from_str(canonical_name, canonical_type)

    # Ensure canonical node exists (may be one of the sources or brand new)
    if canonical_node not in graph.kg:
        canonical_entity = Entity(
            entity_name=canonical_name,
            entity_type=canonical_type,
            description=canonical_desc,
            source_ids=merged_source_ids,
        )
        graph.add_kg_node(canonical_entity)
    else:
        # Update description and source_ids on the existing node
        existing = graph.get_entity_by_node_name(canonical_node)
        merged_source_ids.update(existing.source_ids)
        updated = Entity(
            entity_name=canonical_name,
            entity_type=canonical_type,
            description=canonical_desc or existing.description,
            source_ids=merged_source_ids,
        )
        graph.kg.nodes[canonical_node].update(updated.model_dump())

    # Transfer edges from each source to canonical, then remove source
    for src in source_entities:
        src_node = graph.get_node_name_from_str(src["entity_name"], src["entity_type"])
        if src_node == canonical_node or src_node not in graph.kg:
            continue
        for neighbor in list(graph.kg.neighbors(src_node)):
            if neighbor == canonical_node:
                continue
            edge_data = graph.kg.get_edge_data(src_node, neighbor)
            if not graph.kg.has_edge(canonical_node, neighbor):
                graph.kg.add_edge(canonical_node, neighbor, **edge_data)
        # Update tree2kg
        for tree_id, nodes in graph.tree2kg.items():
            if src_node in nodes:
                nodes.discard(src_node)
                nodes.add(canonical_node)
        graph.kg.remove_node(src_node)

    # Persist source_ids on canonical node
    graph.kg.nodes[canonical_node]["source_ids"] = list(merged_source_ids)
    graph.save_graph()

    canonical_ent = graph.get_entity_by_node_name(canonical_node)
    return [{
        "entity_name": canonical_ent.entity_name,
        "entity_type": canonical_ent.entity_type,
        "description": canonical_ent.description,
        "source_ids": sorted(canonical_ent.source_ids),
        "node_name": canonical_node,
    }]


async def merge_entities(
    tenant_id: str, doc_id: str, config_path: str,
    source_entities: List[dict],
    canonical_name: str, canonical_type: str, canonical_desc: str,
    user_id: str,
) -> List[dict]:
    async with _get_lock(tenant_id, doc_id):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, _merge_sync,
            tenant_id, doc_id, config_path,
            source_entities, canonical_name, canonical_type, canonical_desc,
        )
    await db.log_entity_edit(MONGO_URI, MONGO_DB_PREFIX, tenant_id, {
        "operation": "merge", "doc_id": doc_id, "user_id": user_id,
        "before": source_entities,
        "after": {"entity_name": canonical_name, "entity_type": canonical_type},
    })
    await _schedule_vdb_rebuild(tenant_id, doc_id, config_path)
    return result

