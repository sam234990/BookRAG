from xmlrpc.client import Boolean

from Core.provider.llm import LLM
from Core.provider.vdb import VectorStore
from Core.configs.graph_config import GraphConfig
from Core.provider.embedding import TextEmbeddingProvider
from Core.provider.rerank import TextRerankerProvider
from Core.Index.Graph import Entity, Relationship, Graph
from Core.prompts.kg_prompt import (
    SUMMARIZE_ENTITY,
    DEFAULT_ENTITY_TYPES,
    MergedEntitySchema,
    ENTITY_RESOLUATION_PROMPT,
    ERExtractSel,
    ER_RERANK_INSTRUCTION,
    DESCRIPTION_SYNTHESIS,
)
from Core.utils.utils import truncate_description


from collections import defaultdict
from typing import Optional, List
import os
import json
import shutil
import logging
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)


class KGRefiner:
    """
    A class to refine knowledge graphs (KG).
    Including the basic and advanced refinement methods.
    For the basic refinement, it merges entities with the same name.
    For the advanced refinement, it performs entity resolution.
    """

    # The separator used to merge entity descriptions
    _DESCRIPTION_SEP_ = "<SEP>"

    def __init__(
        self, llm: LLM, graph_config: GraphConfig, graph_index: Graph, save_path: str
    ):
        self.llm = llm
        self.graph_index = graph_index
        self.graph_config = graph_config

        # The following used for advanced refiner
        self.embedder = TextEmbeddingProvider(
            model_name=graph_config.embedding_config.model_name,
            backend=graph_config.embedding_config.backend,
            max_length=graph_config.embedding_config.max_length,
            device=graph_config.embedding_config.device,
            api_base=graph_config.embedding_config.api_base,
        )
        self.reranker = TextRerankerProvider(
            model_name=graph_config.reranker_config.model_name,
            device=graph_config.reranker_config.device,
            max_length=graph_config.reranker_config.max_length,
            backend=graph_config.reranker_config.backend,
            api_base=graph_config.reranker_config.api_base,
        )
        # delete the old vector database if exists
        self.vdb_path = os.path.join(save_path, "kg_vdb")
        if os.path.exists(self.vdb_path):
            log.info(f"Deleting old vector database at {self.vdb_path}")
            # delete this dir
            shutil.rmtree(self.vdb_path)
        self.vdb = VectorStore(
            embedding_model=self.embedder,
            db_path=self.vdb_path,
            collection_name="kg_collection",
        )
        # self.entity_merge_times: dict[str, int] = defaultdict(int)
        self.entity_to_vdb_id: dict[str, str] = defaultdict(str)
        self.entity_alias_map: dict[str, str] = defaultdict(str)

    def close(self) -> None:
        """
        Correctly closes all resources used by the KGRefiner, including the
        Embedder, Reranker
        """
        log.info("Closing KGRefiner and all its resources...")

        # 1. Close the Reranker and release its reference
        if (
            hasattr(self, "reranker")
            and self.reranker
            and hasattr(self.reranker, "close")
        ):
            self.reranker.close()
        self.reranker = None

        # 2. Close the Embedder and release its reference
        if (
            hasattr(self, "embedder")
            and self.embedder
            and hasattr(self.embedder, "close")
        ):
            self.embedder.close()
        self.embedder = None

        # 6. Perform a final garbage collection and empty the CUDA cache
        log.info("Performing final cleanup...")
        gc.collect()

        log.info("✅ KGRefiner resources closed successfully.")

    def get_latest_entity_name(self, node_name: str) -> str:
        if node_name not in self.entity_alias_map.keys():
            raise ValueError(
                f"Entity name '{node_name}' not found in the alias map. "
                "Please ensure the entity has been processed before."
            )
        latest_node_name = self.entity_alias_map[node_name]
        if latest_node_name == node_name:
            return latest_node_name
        else:
            # Recursively find the latest entity name
            return self.get_latest_entity_name(latest_node_name)

    def entity_merge(
        self,
        old_entity: Entity,
        new_entity: Entity,
        merged_to_old_entity: Boolean = False,
    ) -> Entity:
        """
        Merges two entities into one by summarizing their descriptions and updating the graph index.
        Args:
            old_entity (Entity): The old entity to merge.
            new_entity (Entity): The new entity to merge with the old entity.
        Returns:
            Entity: The merged entity with updated description and source IDs.
        """
        # 1. delete old entity from the vector database

        self.delete_entity_from_vdb(old_entity)

        # 2. merge the two entities
        old_node_name = self.graph_index.get_node_name_from_entity(old_entity)
        new_node_name = self.graph_index.get_node_name_from_entity(new_entity)
        if (old_node_name == new_node_name) or merged_to_old_entity:
            # 2.1 if have the same node name, or merged to old entity,
            # Directly merged if the entity name and type are the same
            log.info("merged directly")
            new_description = (
                old_entity.description + self._DESCRIPTION_SEP_ + new_entity.description
            )
            merged_entity = Entity(
                entity_name=old_entity.entity_name,
                entity_type=old_entity.entity_type,
                description=new_description,
                source_ids=set(old_entity.source_ids).union(new_entity.source_ids),
            )
        else:
            # 2.2 if have different node name, use LLM to create new entity
            log.info("merged by LLM summarization")
            old_entity_dict = old_entity.model_dump(exclude={"source_ids"})
            old_entity_dict["description"] = truncate_description(
                old_entity_dict["description"], max_words=200
            )

            new_entity_dict = new_entity.model_dump(exclude={"source_ids"})
            new_entity_dict["description"] = truncate_description(
                new_entity_dict["description"], max_words=200
            )

            prompt = SUMMARIZE_ENTITY.format(
                entity_types=",".join(DEFAULT_ENTITY_TYPES),
                input_json=json.dumps(
                    {
                        "entity_1": old_entity_dict,
                        "entity_2": new_entity_dict,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
            )
            res_entity: MergedEntitySchema = self.llm.get_json_completion(
                prompt=prompt, schema=MergedEntitySchema
            )
            res_entity.entity_name = res_entity.entity_name.lower()
            res_entity.entity_type = res_entity.entity_type.upper()
            res_entity.entity_type = res_entity.entity_type.replace(" ", "_")

            description = (
                old_entity.description + self._DESCRIPTION_SEP_ + new_entity.description
            )

            merged_entity = Entity(
                entity_name=res_entity.entity_name,
                entity_type=res_entity.entity_type,
                description=description,
                source_ids=set(old_entity.source_ids).union(new_entity.source_ids),
            )

            # 2.3 If the llm generated merged entity is another entity (entityC) in the graph,
            # merged entity_c to merged_entity and then update the graph index.
            # delete entity_c from the vdb

            merged_node_name = self.graph_index.get_node_name_from_entity(merged_entity)
            if (
                merged_node_name != old_node_name
                and merged_node_name in self.graph_index.get_all_nodes()
            ):
                # If the merged entity is another entity (entityC) in the graph,
                # merge entityC to old entity and then update the graph index.
                entity_c = self.graph_index.get_entity_by_node_name(merged_node_name)

                log.info(
                    f"Entity '{merged_node_name}' already exists in the graph. "
                    "Merging it with the old entity.\n"
                )
                log.info(
                    f"Old entity: {old_entity.entity_name} ({old_entity.entity_type}), \n"
                    f"New entity: {new_entity.entity_name} ({new_entity.entity_type}), \n"
                    f"Entity C: {entity_c.entity_name} ({entity_c.entity_type})"
                )
                self.delete_entity_from_vdb(entity_c)

                # Merge entityC to old_entity
                self.graph_index.update_entity(
                    old_entity_name=entity_c.entity_name,
                    old_entity_type=entity_c.entity_type,
                    new_entity=old_entity,
                )
                merged_entity.description += (
                    self._DESCRIPTION_SEP_ + entity_c.description
                )
                merged_entity.source_ids = set(merged_entity.source_ids).union(
                    entity_c.source_ids
                )
                # since entity_c is the same as merged_entity, no need to update alias map

        # 3. update the graph index with the merged entity
        # merge old_entity to merged_entity
        self.graph_index.update_entity(
            old_entity_name=old_entity.entity_name,
            old_entity_type=old_entity.entity_type,
            new_entity=merged_entity,
        )

        log.info(
            f"Merged entity '{old_entity.entity_name}' with '{new_entity.entity_name}'. \n"
            f"Old entity type: '{old_entity.entity_type}', \n"
            f"New entity name: '{merged_entity.entity_name}', \n"
            f"New entity type: '{merged_entity.entity_type}', \n"
        )

        # Update the entity alias map
        old_node_name = self.graph_index.get_node_name_from_entity(old_entity)
        new_node_name = self.graph_index.get_node_name_from_entity(new_entity)
        merged_node_name = self.graph_index.get_node_name_from_entity(merged_entity)
        self.entity_alias_map[old_node_name] = merged_node_name
        self.entity_alias_map[new_node_name] = merged_node_name
        self.entity_alias_map[merged_node_name] = merged_node_name

        return merged_entity

    def basic_kg_refiner(
        self, entities: List[Entity], relationships: List[Relationship], source_id: int
    ) -> None:
        """
        Merges entities if they have the same entity name and updates relationships accordingly.
        Args:
            entities (List[Entity]): List of entities to merge.
            relationships (List[Relationship]): List of relationships to update.
            source_id (int): The source ID of this extracted Sub KG.
        """

        # Create a map from original entity name to final entity  (if merged)
        entity_map: dict[str, str] = {}
        add_entity_list = []
        for entity in entities:
            entity_node_name = self.graph_index.get_node_name_from_entity(entity=entity)
            # If the entity is not in the graph index, add it
            # Otherwise, merge it with the existing entity
            if entity_node_name not in self.graph_index.get_all_nodes():
                self.graph_index.add_and_link(
                    tree_node_id=source_id, entities=entity
                )
                entity_map[entity.entity_name] = entity
                add_entity_list.append(entity)
            else:
                # Merge with existing entity
                existing_entity = self.graph_index.get_entity(
                    entity.entity_name, entity.entity_type
                )
                merged_entity = self.entity_merge(existing_entity, entity)
                entity_map[existing_entity.entity_name] = merged_entity
                add_entity_list.append(merged_entity)

        # Add the new entities to the vector database
        self.add_entities_to_vdb(add_entity_list)

        # Update relationships
        for rel in relationships:
            if rel.src_entity_name in entity_map:
                rel.src_entity_name = entity_map[rel.src_entity_name].entity_name
                src_type = entity_map[rel.src_entity_name].entity_type
            if rel.tgt_entity_name in entity_map:
                rel.tgt_entity_name = entity_map[rel.tgt_entity_name].entity_name
                tgt_type = entity_map[rel.tgt_entity_name].entity_type
            self.graph_index.add_kg_edge(rel=rel, src_type=src_type, tgt_type=tgt_type)

    def get_vdb_meta_data(self, entity: Entity) -> dict:
        """
        Generates metadata for the entity to be stored in the vector database.
        Args:
            entity (Entity): The entity to generate metadata for.
        Returns:
            dict: The metadata dictionary without source_ids.
            since vdb does not support list type.
        """
        return {
            "entity_name": entity.entity_name,
            "entity_type": entity.entity_type,
            "description": entity.description,
        }

    def add_entities_to_vdb(self, entities: List[Entity]) -> None:
        """
        Adds a list of entities to the vector database.
        Args:
            entities (List[Entity]): The list of entities to add to the vector database.
        """
        if not entities:
            return

        # deduplicated entities
        entity_map = {}
        for entity in entities:
            node_name = self.graph_index.get_node_name_from_entity(entity)
            if node_name not in entity_map:
                entity_map[node_name] = entity
            else:
                # If the entity already exists, select longer description
                existing_entity = entity_map[node_name]
                if len(entity.description) > len(existing_entity.description):
                    existing_entity.description = entity.description
        
        entities = list(entity_map.values())

        embed_texts = []
        metadatas = []
        for ent in entities:
            node_name = self.graph_index.get_node_name_from_entity(ent)
            if node_name in self.entity_to_vdb_id:
                log.info(
                    f"Entity '{node_name}' already exists in the vector database."
                    "Skipping adding it again."
                )
                continue
            embed_texts.append(node_name)
            metadatas.append(self.get_vdb_meta_data(ent))
        if not embed_texts:
            return
         
        vdbids: List[str] = self.vdb.add_texts(texts=embed_texts, metadatas=metadatas)
        for embed_text, vdbid in zip(embed_texts, vdbids):
            if embed_text in self.entity_to_vdb_id:
                log.warning(
                    f"Entity '{embed_text}' already exists in the vector database. "
                    "Overwriting the existing entry."
                )
            self.entity_to_vdb_id[embed_text] = vdbid

    def delete_entity_from_vdb(self, old_entity: Entity) -> None:
        """
        Deletes an entity from the vector database.
        Args:
            entity (Entity): The entity to delete from the vector database.
        """
        embed_text = self.graph_index.get_node_name_from_entity(old_entity)
        vdbid = self.entity_to_vdb_id.get(embed_text, None)
        if vdbid is not None:
            self.vdb.delete_text_by_ids(ids=[vdbid])
            del self.entity_to_vdb_id[embed_text]
            log.info(f"delete entity {embed_text} from vector database.")
        else:
            log.info(
                f"Entity '{old_entity.entity_name}' with type '{old_entity.entity_type}' "
                f"not found in the vector database. Cannot delete."
            )
            log.info("this may cause add duplicate entities later.")

    def search_similar_entities(
        self, entity: Entity, topk: int = 10, distance_threshold=0.2, mink=1, g=0.6
    ) -> List[Entity]:
        """
        Searches for similar entities in the vector database based on the entity's text information.
        This method is the core method for entity resolution.
        1. First retrieval topk Entities from the vector database.
        2. use the reranker to score these Entities
        3. Gradient-based similar Entities selection.
        4. 1) If all the Entities are not similar enough (With low score), return empty list.
        2) If there are some similar Entities, return the gradient-based truncated list (one or more).
        3) If all Entities are selected, return empty list. This means all Entities are similar enough.

        Args:
            entity (Entity): The entity to search for similar entities.
            topk (int): The number of top similar entities to retrieve from the vector database.
            distance_threshold (float): The maximum distance threshold from the closest entity,
                below threshold, we consider it may have similar entities.
            mink (int): The minimum number of entities to select before gradient-based selection.
            g (float): The gradient factor for selecting additional entities based on their scores.
        Returns:
            List[Entity]: A list of similar entities or empty list if none found.
        """
        embed_text = self.graph_index.get_node_name_from_entity(entity)
        similar_entities = self.vdb.search(embed_text, top_k=topk)
        min_distance = (
            similar_entities[0]["distance"] if similar_entities else float("inf")
        )
        if min_distance > distance_threshold:
            log.info(
                f"No similar entities found for '{entity.entity_name}' with type '{entity.entity_type}'. "
                f"Minimum distance: {min_distance}, threshold: {distance_threshold}."
            )
            return []

        def metadata_str(meta_data: dict):
            description = meta_data.get('description', '')
            
            max_words = 1000
            max_chars = 10000
            
            words = description.split()
            if len(words) > max_words:
                description = " ".join(words[:max_words]) + "..."

            if len(description) > max_chars:
                description = description[:max_chars] + "..."

            entity_str = (
                f"Name: {meta_data.get('entity_name', '')}\n"
                f"Type: {meta_data.get('entity_type', '')}\n"
                f"Description: {description}"
            )
            return entity_str

        similar_entities_str = [
            metadata_str(ent["metadata"]) for ent in similar_entities
        ]

        scores = self.reranker.rerank(
            query=embed_text,
            documents=similar_entities_str,
            instruction=ER_RERANK_INSTRUCTION,
        )
        self.reranker.clean_cache()

        ranked_results = sorted(
            zip(similar_entities, scores), key=lambda x: x[1], reverse=True
        )

        # 4.1 max score < 0.5 not similar enough, return empty list
        if not ranked_results or ranked_results[0][1] < 0.5:
            return []

        # 4.2 gradient-based selection
        # add first min_k entities to the selection list
        sel_entities = ranked_results[:mink]
        score_remain = sel_entities[-1][1]  # the score of the last selected entity

        # add the remaining entities based on the gradient-based selection
        for ent, score in ranked_results[mink:]:
            if score >= score_remain * g:
                sel_entities.append((ent, score))
                score_remain = score
            else:
                break

        if len(sel_entities) == ranked_results:
            # 4.3 If all entities are selected, return empty list
            return []

        res_entities = []
        for ent, _ in sel_entities:
            entity_name = ent["metadata"].get("entity_name", "")
            entity_type = ent["metadata"].get("entity_type", "")
            res_entities.append(self.graph_index.get_entity(entity_name, entity_type))
        return res_entities

    def _prepare_selection_input(
        self, new_entity: Entity, similar_entities: List[Entity]
    ) -> str:
        """Formats the entities into the JSON structure required by the prompt."""

        # Give each similar entity a temporary ID (index)
        candidates_with_ids = []
        for i, entity in enumerate(similar_entities):
            entity_dict = entity.model_dump(exclude={"source_ids"})
            if "description" in entity_dict and entity_dict["description"]:
                entity_dict["description"] = truncate_description(
                    entity_dict["description"]
                )
            entity_dict["id"] = i
            candidates_with_ids.append(entity_dict)

        input_data = {
            "new_entity": new_entity.model_dump(exclude={"source_ids"}),
            "candidate_entities": candidates_with_ids,
        }

        return json.dumps(input_data, indent=2, ensure_ascii=False)

    def er_selection_by_llm(
        self, new_entity: Entity, similar_entities: List[Entity]
    ) -> Optional[Entity]:
        # 1. Prepare the input for the LLM
        input_json_str = self._prepare_selection_input(new_entity, similar_entities)
        prompt = ENTITY_RESOLUATION_PROMPT.format(input_json=input_json_str)

        # 2. Call the LLM
        try:
            res: ERExtractSel = self.llm.get_json_completion(
                prompt=prompt, schema=ERExtractSel
            )
        except Exception as e:
            log.error(f"LLM call failed: {e}")
            return None

        # 3. Parse the LLM response
        select_id = res.select_id

        # 4. Return the result
        if select_id == -1:
            log.info(
                f"LLM did not select any similar entity for the entity:\n {new_entity.entity_name} "
            )
            log.info(f"Reason:\n {res.explanation}")
            return None

        if 0 <= select_id < len(similar_entities):
            # Log the selection and reason
            log.info(
                f"LLM selected entity ID: {select_id}, " f"Reason: {res.explanation}"
            )
            # Log the new entity and the selected similar entity
            log.info("New Entity Info:")
            log.info(
                f"Entity Name: {new_entity.entity_name}, Entity Type: {new_entity.entity_type}"
            )
            log.info(f"LLM selected Entity Info:")
            log.info(
                f"Entity Name: {similar_entities[select_id].entity_name}, Entity Type: {similar_entities[select_id].entity_type}"
            )

            return similar_entities[select_id]
        else:
            print(f"Warning: LLM returned an out-of-bounds ID: {select_id}")
            return None

    def entity_resolution(self, new_entity: Entity) -> Entity:
        """
        Resolves the new entity by comparing it with similar entities.
        Merge the new entity with the most similar one if they are true duplicated entity.

        Args:
            new_entity (Entity): The new entity to resolve.
        Returns:
            Entity: The resolved entity, which may be a merged entity or the new entity itself.
        """

        # 1.1 the same entity name and entity type, merge them directly
        node_name = self.graph_index.get_node_name_from_entity(entity=new_entity)
        if node_name in self.graph_index.get_all_nodes():
            # If the entity already exists in the graph index with the same type, merge them
            existing_entity = self.graph_index.get_entity(
                new_entity.entity_name, new_entity.entity_type
            )

            merged_entity = self.entity_merge(existing_entity, new_entity)
            return merged_entity

        # 1.2 If the entity have a same name with existing entity, but merged before.
        # Merged them directly
        if node_name in self.entity_alias_map.keys():
            latest_entity_name = self.get_latest_entity_name(node_name=node_name)
            log.info(
                f"Entity '{new_entity.entity_name}' with type '{new_entity.entity_type}' "
                f"has been merged before. Merging with the existing entity."
                f"Latest node: {latest_entity_name}"
            )
            existing_entity = self.graph_index.get_entity_by_node_name(
                latest_entity_name
            )

            merged_entity = self.entity_merge(
                existing_entity, new_entity, merged_to_old_entity=True
            )

            return merged_entity

        # 2. search similar entities in the vector database
        similar_entities = self.search_similar_entities(new_entity)
        if len(new_entity.source_ids) != 1:
            raise ValueError(
                f"Expected exactly one source_id, but found {len(new_entity.source_ids)}."
            )
        
        source_id = next(iter(new_entity.source_ids))
        
        if len(similar_entities) == 0:
            # 2.1 No similar entities found, add the new entity directly
            self.graph_index.add_and_link(
                tree_node_id=source_id, entities=new_entity
            )
            return new_entity

        # 2.2 If similar entities are found, use the LLM to determine if exist one of them is the same entity as the new one.
        sel_existing_entity = self.er_selection_by_llm(
            new_entity=new_entity, similar_entities=similar_entities
        )
        if sel_existing_entity is None:
            # If no similar entity is selected, add the new entity directly
            self.graph_index.add_and_link(
                tree_node_id=source_id, entities=new_entity
            )
            return new_entity
        else:
            # If a similar entity is selected, merge the new entity with it
            merged_entity: Entity = self.entity_merge(sel_existing_entity, new_entity)

            return merged_entity

    def process_unknown_entities(
        self, unknown_entities: List[Entity], entity_map: dict[str, Entity]
    ) -> dict[str, Entity]:
        log.info(f"Processing unknown entities, length: {len(unknown_entities)}")
        if unknown_entities:
            unknown_vdb_entities = []
            for entity in unknown_entities:
                # Perform entity resolution for unknown entities
                old_entity_name = entity.entity_name
                new_entity: Entity = self.entity_resolution(entity)
                entity_map[old_entity_name] = new_entity
                unknown_vdb_entities.append(new_entity)
            # Add the resolved unknown entities to the vector database
            self.add_entities_to_vdb(unknown_vdb_entities)
        return entity_map

    def process_relationships(
        self, relationships: List[Relationship], entity_map: dict[str, Entity]
    ) -> None:
        """
        Processes relationships by updating source and target entity names based on the entity map.
        And adds them to the graph index.
        Args:
            relationships (List[Relationship]): List of relationships to process.
            entity_map (dict[str, Entity]): Map of old entity names to new entities.
        """
        for k, v in entity_map.items():
            node_name = self.graph_index.get_node_name_from_entity(v)
            if node_name not in self.graph_index.get_all_nodes():
                new_node_name = self.get_latest_entity_name(node_name=node_name)
                entity_map[k] = self.graph_index.get_entity_by_node_name(new_node_name)
                log.info(
                    f"Entity '{v.entity_name}' with type '{v.entity_type}' not found in the graph index. "
                    f"Using the latest entity '{new_node_name}' instead."
                )

        for rel in relationships:
            old_src_name = rel.src_entity_name
            old_tgt_name = rel.tgt_entity_name
            src_type = None
            tgt_type = None
            if old_src_name in entity_map:
                rel.src_entity_name = entity_map[old_src_name].entity_name
                src_type = entity_map[old_src_name].entity_type
            if old_tgt_name in entity_map:
                rel.tgt_entity_name = entity_map[old_tgt_name].entity_name
                tgt_type = entity_map[old_tgt_name].entity_type
            if src_type is None or tgt_type is None:
                log.info(
                    f"Relationship {rel} has missing entity types. "
                    "Skipping this relationship."
                )
                continue
            else:
                self.graph_index.add_kg_edge(
                    rel=rel, src_type=src_type, tgt_type=tgt_type
                )

    def _debug_check_num(self):
        num_node_graph = len(self.graph_index.kg.nodes())
        num_node_vdb = self.vdb.collection.count()
        if num_node_graph != num_node_vdb:
            log.warning(
                f"Number of nodes in the graph index ({num_node_graph}) "
                f"does not match the number of nodes in the vector database ({num_node_vdb})."
            )
            print("warning here")
        else:
            log.info(
                f"graph and vdb contain the same number of nodes: {num_node_graph}."
            )

    def advanced_kg_refiner(
        self, entities: List[Entity], relationships: List[Relationship], source_id: int
    ) -> None:
        """
        Refines the knowledge graph by advanced entity resolution and relationship updates.
        Args:
            entities (List[Entity]): List of entities to refine.
            relationships (List[Relationship]): List of relationships to update.
            source_id (int): The source ID of this extracted tree node.
        """
        log.info(
            f"--------------------\n"
            f"Starting advanced knowledge graph refinement for source ID: {source_id}\n"
            f"with {len(entities)} entities and {len(relationships)} relationships."
        )

        # map the old entity name to the new entity name after resolution
        entity_map: dict[str, Entity] = {}

        # 1. for the first time to refine the KG, the vector database and graph index are initialized.
        if self.vdb.collection.count() <= 10:
            # If the vector database is empty or has very few entities, we can skip entity resolution.
            # Not entity resolution for normal entities.

            add_entities = []
            unknown_entities = []
            for entity in entities:
                if entity.entity_type != "UNKNOWN":
                    # For normal entities, we can add them directly to the vector database and graph index.
                    add_entities.append(entity)
                    entity_map[entity.entity_name] = entity
                else:
                    unknown_entities.append(entity)

            # add to vdb and graph
            self.add_entities_to_vdb(entities)
            self.graph_index.add_and_link(
                tree_node_id=source_id, entities=entities
            )

            # For unknown entities, we need to resoluation them
            entity_map = self.process_unknown_entities(
                unknown_entities=unknown_entities, entity_map=entity_map
            )

            # Update relationships based on the entity map
            self.process_relationships(relationships, entity_map)
        else:
            # 2. For each entity, perform resolution and update the graph index.
            new_entity_list = []
            unknown_entities = []
            for entity in entities:
                if entity.entity_type == "UNKNOWN":
                    # 2.1 for unknown entity type, perform resolution
                    # For unknown entities, we need to resolve them later.
                    unknown_entities.append(entity)
                    continue

                # 2.2 for other entity types, perform resolution
                old_entity_name = entity.entity_name
                new_entity: Entity = self.entity_resolution(entity)
                entity_map[old_entity_name] = new_entity
                new_entity_list.append(new_entity)

            # 2.3 Add the resolved entities to the vector database
            # Since the ER should not be performed within the same chunk
            # The new entities should not be in the vector database yet.
            self.add_entities_to_vdb(new_entity_list)

            # 2.4 Address the unknown entities
            entity_map = self.process_unknown_entities(
                unknown_entities=unknown_entities, entity_map=entity_map
            )

            # 3. Update relationships based on the resolved entities
            self.process_relationships(
                relationships=relationships, entity_map=entity_map
            )

        # for debug check the number of nodes in graph and vdb
        self._debug_check_num()

    def refine_entity_description(self, entity: Entity) -> Entity:
        # use LLM to refine the entity description
        # update the graph
        # delete the old entity from the vector database, insert new one later
        log.info(
            f"Refining entity description for {entity.entity_name} of type {entity.entity_type}."
        )
        json_entity = entity.model_dump(exclude={"source_ids"})
        prompt = DESCRIPTION_SYNTHESIS.format(
            input_json=json.dumps(json_entity, indent=2, ensure_ascii=False)
        )
        try:
            refined_description = self.llm.get_completion(
                prompt=prompt, json_response=False
            )
            if not refined_description:
                log.warning(
                    f"LLM returned an empty description for entity {entity.entity_name}."
                )
                return entity
            else:
                # Update the entity description
                entity.description = refined_description
                # Update the graph index with the new description
                self.graph_index.update_entity(
                    old_entity_name=entity.entity_name,
                    old_entity_type=entity.entity_type,
                    new_entity=entity,
                )
                # Delete the old entity from the vector database
                self.delete_entity_from_vdb(entity)
                log.info(
                    f"Entity {entity.entity_name} description refined successfully."
                )
                return entity
        except Exception as e:
            log.error(
                f"Failed to refine entity description for {entity.entity_name}: {e}"
            )
            return entity

    def refine_entities(self):
        merged_entity_set = set()
        need_refine_entities = []
        for node_name in self.entity_alias_map.keys():
            latest_entity_name = self.get_latest_entity_name(node_name)
            if latest_entity_name not in merged_entity_set:
                merged_entity_set.add(latest_entity_name)
                # Get the entity from the graph index
                entity = self.graph_index.get_entity_by_node_name(latest_entity_name)

                # check the sep in the description
                if self._DESCRIPTION_SEP_ in entity.description:
                    # If the description contains the separator, we need to refine it
                    need_refine_entities.append(entity)
                else:
                    # If the description does not contain the separator, we can skip it
                    continue

        if not need_refine_entities:
            log.info("No entities need to be refined.")
            return

        log.info(f"Found {len(need_refine_entities)} entities that need to be refined.")

        # parallel processing of entity refinement
        add_entities = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self.refine_entity_description, entity): entity
                for entity in need_refine_entities
            }
            for future in as_completed(futures):
                entity = futures[future]
                try:
                    refined_entity = future.result()
                    add_entities.append(refined_entity)
                except Exception as e:
                    log.error(f"Failed to refine entity {entity.entity_name}: {e}")
                    add_entities.append(entity)

        # Add the refined entities to the vector database
        self.add_entities_to_vdb(add_entities)
        log.info(
            f"Refined {len(add_entities)} entities and added them to the vector database."
        )
        self._debug_check_num()
        return

    def refine_relation(self):
        # delete self loop in graph index
        self.graph_index.remove_self_loops()