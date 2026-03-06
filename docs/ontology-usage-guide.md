## Ontology usage guide

Use ontology entities when you want BookRAG to normalize extracted mentions onto stable canonical entities during indexing and retrieval.

### Inline ontology config

Set `ontology.enabled: true` and define entities directly in the config file.

- `ontology_id`: stable canonical identifier
- `canonical_name`: normalized entity name stored in the graph
- `entity_type`: type used during matching
- `aliases`: alternate spellings and surface forms
- `keywords`: optional matching hints for domain terminology

Example fields:

- `mapping_threshold`: stricter values reduce false matches
- `allow_provisional_entities`: keep unmatched entities in the graph when `true`
- `use_query_resolution`: resolve query mentions to ontology-backed graph nodes first

### File-backed ontology config

You can also point `ontology.path` at a YAML or JSON file. Relative paths are resolved from the config file location. Inline entities and file-backed entities are merged by `ontology_id`.

### Phase 3 tenant/global resolution

Phase 3 is now controlled by `entity_resolution` config:

- `enabled`: turns tenant/global canonical resolution on or off
- `similarity_threshold`: vector similarity gate for reuse of an existing tenant-global entity
- `top_k`: number of nearest global candidates to inspect
- `global_vdb_dir`: directory for the tenant-global ChromaDB store
- `collection_name`: ChromaDB collection name for global entities
- `canonical_only`: only export ontology-backed/canonical entities when `true`
- `sync_to_global_graph`: also sync to tenant-global FalkorDB when `true`

Relative `global_vdb_dir` values are resolved from the config file location.

### Recommended starting config

- keep `ontology.enabled: true`
- start with `mapping_threshold: 1.0`
- keep `allow_provisional_entities: true`
- keep `entity_resolution.enabled: false` until you want cross-document tenant normalization
- once enabled, start with `canonical_only: true` if your ontology is strong and curated

### Current limitation

Phase 3 currently uses vector similarity plus canonical metadata persistence. It is intentionally conservative and does not yet implement a full mention-node merge model in the tenant-global graph.