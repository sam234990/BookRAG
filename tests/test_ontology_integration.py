import json

import yaml

from Core.Index.Graph import Entity, Graph, Relationship
from Core.configs.ontology_config import OntologyConfig
from Core.configs.system_config import load_system_config
from Core.utils.ontology_utils import (
    align_entities_to_ontology,
    find_best_graph_ontology_node,
)


def test_load_system_config_resolves_relative_ontology_path_and_merges_entities(tmp_path):
    ontology_path = tmp_path / "ontology.yaml"
    ontology_path.write_text(
        yaml.safe_dump(
            {
                "entities": [
                    {
                        "ontology_id": "product:file-backed",
                        "canonical_name": "file backed product",
                        "entity_type": "PRODUCT",
                        "description": "Loaded from ontology file.",
                        "aliases": ["fb product"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "mineru": {
                    "backend": "vlm-sglang-client",
                    "method": "vlm",
                    "lang": "en",
                },
                "rag": {"strategy": "gbc"},
                "ontology": {
                    "enabled": True,
                    "path": "ontology.yaml",
                    "entities": [
                        {
                            "ontology_id": "product:inline",
                            "canonical_name": "inline product",
                            "entity_type": "PRODUCT",
                            "description": "Inline ontology entity.",
                            "aliases": ["inline alias"],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_system_config(str(config_path))

    assert cfg.ontology.path == str(ontology_path.resolve())
    assert {entity.entity_id for entity in cfg.ontology.entities} == {
        "product:inline",
        "product:file-backed",
    }


def test_align_entities_to_ontology_maps_entities_and_relationships():
    ontology_cfg = OntologyConfig(
        enabled=True,
        entities=[
            {
                "ontology_id": "product:bookrag",
                "canonical_name": "bookrag",
                "entity_type": "PRODUCT",
                "description": "The canonical BookRAG product entity.",
                "aliases": ["book rag"],
            }
        ],
    )
    entities = [
        Entity(entity_name="Book Rag", entity_type="product", description="Mentioned in text."),
        Entity(entity_name="Retriever Engine", entity_type="system", description="Local component."),
    ]
    relationships = [
        Relationship(
            src_entity_name="Book Rag",
            tgt_entity_name="Retriever Engine",
            relation_name="uses",
        )
    ]

    aligned_entities, aligned_relationships = align_entities_to_ontology(
        entities, relationships, ontology_cfg
    )

    canonical = next(entity for entity in aligned_entities if entity.entity_role == "canonical")
    provisional = next(entity for entity in aligned_entities if entity.entity_role == "provisional")
    assert canonical.entity_name == "bookrag"
    assert canonical.canonical_id == "product:bookrag"
    assert "book rag" in canonical.aliases
    assert provisional.entity_name == "retriever engine"
    assert aligned_relationships[0].src_entity_name == "bookrag"
    assert aligned_relationships[0].tgt_entity_name == "retriever engine"


def test_align_entities_to_ontology_drops_unmatched_entities_when_provisional_disabled():
    ontology_cfg = OntologyConfig(
        enabled=True,
        allow_provisional_entities=False,
        entities=[
            {
                "ontology_id": "product:bookrag",
                "canonical_name": "bookrag",
                "entity_type": "PRODUCT",
                "description": "The canonical BookRAG product entity.",
                "aliases": ["book rag"],
            }
        ],
    )
    entities = [
        Entity(entity_name="Book Rag", entity_type="product"),
        Entity(entity_name="Unknown System", entity_type="system"),
    ]
    relationships = [
        Relationship(
            src_entity_name="Book Rag",
            tgt_entity_name="Unknown System",
            relation_name="uses",
        )
    ]

    aligned_entities, aligned_relationships = align_entities_to_ontology(
        entities, relationships, ontology_cfg
    )

    assert [entity.entity_name for entity in aligned_entities] == ["bookrag"]
    assert aligned_relationships == []


def test_graph_update_entity_rewrites_edge_payload_names_and_tree_links(tmp_path):
    graph = Graph(save_path=str(tmp_path))
    old_entity = Entity(entity_name="book rag", entity_type="PRODUCT", source_ids={1})
    other_entity = Entity(entity_name="retriever", entity_type="SYSTEM", source_ids={2})
    graph.add_and_link(tree_node_id=1, entities=old_entity)
    graph.add_and_link(tree_node_id=2, entities=other_entity)
    graph.add_kg_edge(
        Relationship(
            src_entity_name="book rag",
            tgt_entity_name="retriever",
            relation_name="uses",
        ),
        src_type="PRODUCT",
        tgt_type="SYSTEM",
    )

    renamed_entity = Entity(
        entity_name="bookrag",
        entity_type="PRODUCT",
        entity_role="canonical",
        canonical_id="product:bookrag",
        source_ids={1},
    )
    graph.update_entity("book rag", "PRODUCT", renamed_entity)

    new_node_name = graph.get_node_name_from_entity(renamed_entity)
    other_node_name = graph.get_node_name_from_entity(other_entity)
    edge_data = graph.kg.get_edge_data(new_node_name, other_node_name)
    assert edge_data["src_entity_name"] == "bookrag"
    assert graph.node_name_to_tree_nodes(new_node_name) == [1]


def test_graph_metadata_round_trip_and_ontology_lookup(tmp_path):
    graph = Graph(save_path=str(tmp_path))
    entity = Entity(
        entity_name="bookrag",
        entity_type="PRODUCT",
        description="Canonical product entity.",
        entity_id="product:bookrag",
        canonical_id="product:bookrag",
        entity_role="canonical",
        aliases=["bookrag", "book rag"],
        mapping_confidence=1.0,
        ontology_source="config",
        source_ids={7},
    )
    graph.add_and_link(tree_node_id=7, entities=entity)
    graph.save_graph()

    loaded = Graph.load_from_dir(str(tmp_path))
    loaded_entity = loaded.get_entity("bookrag", "PRODUCT")
    metadata = loaded_entity.to_vdb_metadata()

    assert loaded_entity.canonical_id == "product:bookrag"
    assert json.loads(metadata["aliases_json"]) == ["bookrag", "book rag"]
    assert find_best_graph_ontology_node(loaded, "book rag", "product", threshold=1.0) == (
        loaded.get_node_name_from_entity(loaded_entity)
    )