from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Iterable, Optional, Tuple

if TYPE_CHECKING:
    from Core.Index.Graph import Graph


def normalize_entity_name(value: str) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def normalize_entity_type(value: str) -> str:
    return str(value or "").strip().upper().replace(" ", "_")


def dedupe_terms(values: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        normalized = normalize_entity_name(value)
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def types_compatible(left: str, right: str) -> bool:
    left_norm = normalize_entity_type(left)
    right_norm = normalize_entity_type(right)
    if not left_norm or not right_norm:
        return True
    if left_norm in {"QUESTION", "UNKNOWN"}:
        return True
    return left_norm == right_norm


def entity_name_similarity(left: str, right: str) -> float:
    left_norm = normalize_entity_name(left)
    right_norm = normalize_entity_name(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def find_best_ontology_match(
    entity, ontology_cfg
) -> Tuple[Optional[object], float]:
    if not getattr(ontology_cfg, "enabled", False):
        return None, 0.0

    best_match = None
    best_score = 0.0
    for candidate in getattr(ontology_cfg, "entities", []):
        if getattr(candidate, "status", "active") == "deprecated":
            continue
        if not types_compatible(entity.entity_type, candidate.entity_type):
            continue
        candidate_terms = dedupe_terms(
            [candidate.entity_name, *candidate.aliases, *getattr(candidate, "keywords", [])]
        )
        score = max(entity_name_similarity(entity.entity_name, alias) for alias in candidate_terms)
        if score > best_score:
            best_match = candidate
            best_score = score

    threshold = getattr(ontology_cfg, "mapping_threshold", 1.0)
    if best_match is None or best_score < threshold:
        return None, best_score
    return best_match, best_score


def align_entities_to_ontology(entities, relationships, ontology_cfg):
    if not getattr(ontology_cfg, "enabled", False) or not getattr(ontology_cfg, "entities", None):
        return entities, relationships

    aligned_entities = []
    original_name_map: dict[str, str] = {}
    allow_provisional_entities = getattr(ontology_cfg, "allow_provisional_entities", True)
    for entity in entities:
        original_name = entity.entity_name
        matched_entity, confidence = find_best_ontology_match(entity, ontology_cfg)
        if matched_entity is not None:
            description = matched_entity.description or entity.description
            if matched_entity.description and entity.description:
                entity_desc = entity.description.strip()
                onto_desc = matched_entity.description.strip()
                if entity_desc and entity_desc not in onto_desc:
                    description = f"{onto_desc}\n\nMention detail: {entity_desc}"
            aligned = entity.model_copy(
                update={
                    "entity_name": matched_entity.entity_name,
                    "entity_type": matched_entity.entity_type,
                    "description": description,
                    "entity_id": matched_entity.entity_id,
                    "canonical_id": matched_entity.entity_id,
                    "entity_role": "canonical",
                    "aliases": dedupe_terms(
                        [entity.entity_name, matched_entity.entity_name, *matched_entity.aliases]
                    ),
                    "mapping_confidence": confidence,
                    "ontology_source": matched_entity.ontology_source,
                }
            )
        elif allow_provisional_entities:
            aligned = entity.model_copy(
                update={
                    "entity_name": normalize_entity_name(entity.entity_name),
                    "entity_type": normalize_entity_type(entity.entity_type),
                    "entity_role": entity.entity_role or "provisional",
                    "aliases": dedupe_terms([entity.entity_name, *entity.aliases]),
                    "mapping_confidence": entity.mapping_confidence,
                }
            )
        else:
            continue
        original_name_map[original_name] = aligned.entity_name
        aligned_entities.append(aligned)

    aligned_relationships = []
    for relationship in relationships:
        if (
            relationship.src_entity_name not in original_name_map
            or relationship.tgt_entity_name not in original_name_map
        ):
            continue
        aligned_relationship = relationship.model_copy(deep=True)
        aligned_relationship.src_entity_name = original_name_map.get(
            relationship.src_entity_name, relationship.src_entity_name
        )
        aligned_relationship.tgt_entity_name = original_name_map.get(
            relationship.tgt_entity_name, relationship.tgt_entity_name
        )
        aligned_relationships.append(aligned_relationship)

    return aligned_entities, aligned_relationships


def find_best_graph_ontology_node(
    graph: "Graph", entity_name: str, entity_type: str, threshold: float = 1.0
) -> Optional[str]:
    best_node_name = None
    best_score = 0.0
    for node_name in graph.get_all_nodes():
        entity = graph.get_entity_by_node_name(node_name)
        if entity.entity_role != "canonical" and not entity.canonical_id:
            continue
        if not types_compatible(entity_type, entity.entity_type):
            continue
        score = max(
            entity_name_similarity(entity_name, alias)
            for alias in dedupe_terms([entity.entity_name, *entity.aliases])
        )
        if score > best_score:
            best_score = score
            best_node_name = node_name
    if best_score < threshold:
        return None
    return best_node_name