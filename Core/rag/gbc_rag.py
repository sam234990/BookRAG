from collections import defaultdict
from typing import Any, List, Tuple, Dict, Optional

from regex import F

from Core.Index.Tree import TreeNode, NodeType
from Core.rag.base_rag import BaseRAG
from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.provider.rerank import TextRerankerProvider
from Core.provider.embedding import MMRerankerProvider
from Core.configs.rag.gbc_config import GBCRAGConfig
from Core.Index.GBCIndex import GBC
from Core.prompts.gbc_prompt import (
    LLM_EXPANSION_SELECT_PROMPT,
    QuestionEntity,
    QuestionEntityExtraction,
    QUESTION_ENT_PROMPT,
    QUESTION_ENTITY_TYPES,
    SecEXPSelection,
)
from Core.Index.Graph import Entity
from Core.rag.gbc_answer import AnswerAgent
from Core.rag.gbc_plan import TaskPlanner, PlanResult
from Core.rag.gbc_retrieval import Retriever

from Core.rag.gbc_utils import (
    GBCRAGContext,
    SubStep,
    filter_tree_nodes,
)


import json
import networkx as nx

import logging

log = logging.getLogger(__name__)


class GBCRAG(BaseRAG):
    """
    GBC RAG (Graph-Based Contextual Retrieval Augmented Generation) class.
    This class is designed to handle the retrieval and generation of responses
    based on a graph-based context.
    """

    def __init__(
        self,
        llm: LLM,
        vlm: VLM,
        config: GBCRAGConfig,
        gbc_index: GBC,
    ):
        super().__init__(
            llm,
            name="GBC RAG",
            description="Graph-Based Contextual Retrieval Augmented Generation",
        )
        self.vlm = vlm
        self.cfg = config
        self.varient = self.cfg.varient
        if not gbc_index:
            raise ValueError("GBC index must be provided for GBCRAG.")
        self.gbc_index = gbc_index
        self.embedder = self.gbc_index.embedder if self.gbc_index else None
        self.reranker = TextRerankerProvider(
            model_name=self.cfg.reranker_config.model_name,
            max_length=self.cfg.reranker_config.max_length,
            device=self.cfg.reranker_config.device,
            backend=self.cfg.reranker_config.backend,
            api_base=self.cfg.reranker_config.api_base,
        )
        # GBC RAG config
        self.threshold_e = self.cfg.sim_threshold_e
        self.select_depth = self.cfg.select_depth
        self.max_retry = self.cfg.max_retry

        # Agents
        self.planner = TaskPlanner(llm=self.llm)
        self.answer = AnswerAgent(llm=self.llm, vlm=self.vlm)
        self.retriever = Retriever(
            varient=self.varient,
            reranker=self.reranker,
            # mm_reranker=self.mm_reranker,
            embedder=self.embedder,
            alpha=self.cfg.alpha,
            topk_ent=self.cfg.topk_ent,
            x_percentile=self.cfg.x_percentile,
            topk=self.cfg.topk,
        )

    def _get_entity_embed_text(self, entity: QuestionEntity) -> str:
        return f"Name: {entity.entity_name}\nType: {entity.entity_type}"

    def _entity_map(
        self, entities: List[str], force_one: bool = False
    ) -> Dict[str, List[str]]:
        """
        Maps entities to their corresponding IDs in the GBC index.
        Use vdb to find the entity in GBC index.
        """
        entities_str = [self._get_entity_embed_text(entity) for entity in entities]
        Qent_GBCent_map = defaultdict(list)
        res_list = []
        for ent_str in entities_str:
            query_res = self.gbc_index.entity_vdb.search(query_text=ent_str, top_k=2)
            min_distance = query_res[0]["distance"] if query_res else float("inf")
            retrieve_name = query_res[0]["metadata"].get("entity_name")
            retrieve_type = query_res[0]["metadata"].get("entity_type")
            node_name = self.gbc_index.GraphIndex.get_node_name_from_str(
                retrieve_name, retrieve_type
            )
            if min_distance < self.threshold_e:
                Qent_GBCent_map[ent_str].append(node_name)
                log.info(f"Entity '{ent_str}' mapped to GBC entity: {node_name}")
            else:
                res_list.append((ent_str, node_name, min_distance))

        if force_one and len(Qent_GBCent_map) == 0 and len(res_list) > 0:
            # force map the closest entity if no entity is mapped
            res_list = sorted(res_list, key=lambda x: x[2])
            ent_str, node_name, min_distance = res_list[0]
            Qent_GBCent_map[ent_str].append(node_name)
            log.info(f"Force map entity '{ent_str}' to GBC entity: {node_name}")

        return Qent_GBCent_map

    def _get_query_entity(self, query: str) -> Dict[str, List[str]]:
        """
        Get the entity mapping for the query.
        """

        # 1. retrieval relevent entities from the query
        retrieval_ents = self.gbc_index.entity_vdb.search(query_text=query, top_k=5)
        retrieval_node_names = set()
        retrieval_nodes = []
        for ent_info in retrieval_ents:
            ent_name = ent_info["metadata"].get("entity_name")
            ent_type = ent_info["metadata"].get("entity_type")
            node_dict = {
                "entity_name": ent_name,
                "entity_type": ent_type,
            }
            node_name = self.gbc_index.GraphIndex.get_node_name_from_str(
                ent_name, ent_type
            )
            if node_name not in retrieval_node_names:
                retrieval_node_names.add(node_name)
                retrieval_nodes.append(node_dict)

        # 2. llm generate and select entities from the query
        prompt = QUESTION_ENT_PROMPT.format(
            input_text=query,
            entity_types=", ".join(QUESTION_ENTITY_TYPES),
            retrieved_entities=json.dumps(retrieval_nodes, ensure_ascii=False),
        )
        res_entities = []
        try:
            res: QuestionEntityExtraction = self.llm.get_json_completion(
                prompt, QuestionEntityExtraction
            )
            if res and res.entities:
                res_entities = res.entities
                entities_name = [entity.entity_name for entity in res_entities]
                log.info(f"Extracted entities: {entities_name}")
            else:
                log.info("No entities extracted from the query.")

        except Exception as e:
            log.error(f"Error during entity extraction: {e}")

        if len(res_entities) == 0:
            # use the retrieval entities if no entity is extracted by llm
            log.info("Use the question as the entity.")
            res_entities = [Entity(entity_name=query, entity_type="Question")]

        Qent_GBCent_map = defaultdict(list)
        remain_ents = []
        for res_ent in res_entities:
            res_ent.entity_name = res_ent.entity_name.lower()
            res_ent.entity_type = res_ent.entity_type.upper()
            res_ent.entity_type = res_ent.entity_type.replace(" ", "_")
            ent_node_name = self.gbc_index.GraphIndex.get_node_name_from_entity(res_ent)
            if ent_node_name in retrieval_node_names:
                Qent_GBCent_map[ent_node_name].append(ent_node_name)
                log.info(
                    f"Entity '{ent_node_name}' mapped to GBC entity: {ent_node_name}"
                )
            else:
                remain_ents.append(res_ent)

        should_force_one = (len(Qent_GBCent_map) == 0)
        if remain_ents:
            remain_map = self._entity_map(remain_ents, force_one=should_force_one)
            for k, v in remain_map.items():
                Qent_GBCent_map[k].extend(v)

        return Qent_GBCent_map

    def link_tree_node(self, entities_map: Dict[str, List[str]]) -> List[dict]:
        """
        Get the tree nodes for the given entities.
        """
        tree_node_cnt = defaultdict(list)
        all_map_nodenames = set()
        for ent_list in entities_map.values():
            for ent in ent_list:
                all_map_nodenames.add(ent)
        all_map_nodenames = list(all_map_nodenames)
        if not all_map_nodenames:
            log.warning("No entities found in the mapping.")
            return []

        for node_name in all_map_nodenames:
            tree_node_set = self.gbc_index.GraphIndex.NodeName2TreeNodes(node_name)
            for node_id in tree_node_set:
                tree_node_cnt[node_id].append(node_name)

        tree_nodes = [
            {
                "index_id": node_id,
                "map_cnt": len(link_ents),
                "linked_entities": link_ents,
            }
            for node_id, link_ents in sorted(
                tree_node_cnt.items(), key=lambda x: len(x[1]), reverse=True
            )
        ]

        if not tree_nodes:
            log.warning("No tree nodes found for the given entities.")
            return []

        log.info(f"Retrieved {len(tree_nodes)} tree nodes based on entity mapping.")
        return tree_nodes

    def link_section(self, tree_nodes: List[dict]) -> Dict[int, List[str]]:
        """
        Get the linked section TreeNode ids from the tree nodes.
        given the tree nodes, get the linked section TreeNode ids (specific depth).
        return the Dict: section_id --> [linked_entity1, linked_entity2, ...]
        """
        sec_entity_map = defaultdict(list)
        for node in tree_nodes:
            node_idx = node["index_id"]
            ancestor = self.gbc_index.TreeIndex.get_ancestor_at_depth(
                node_idx, self.select_depth
            )
            ancestor_idx = ancestor.index_id if ancestor else None
            node_ents = node["linked_entities"]
            if ancestor_idx:
                sec_entity_map[ancestor_idx].extend(node_ents)

        for sec_id, val in sec_entity_map.items():
            sec_entity_map[sec_id] = list(set(val))

        log.info(
            f"Found {len(sec_entity_map)} linked sections at depth {self.select_depth}."
        )
        return sec_entity_map

    def prep_SecSel_prompt(
        self,
        query,
        link_nodes: List[TreeNode] = None,
        remain_nodes: List[TreeNode] = None,
        sec_entity_map: Dict[int, List[str]] = None,
    ) -> str:
        """
        Prepare the prompt for section selection.
        This method should be implemented to prepare the prompt
        """

        def prep_nodes_json(
            nodes: List[TreeNode], sec_entity_map: Dict[int, List[str]] = None
        ) -> str:
            node_infos = []
            for node in nodes:
                sec_idx = node.index_id
                section_title = node.meta_info.content
                sec_path = self.gbc_index.TreeIndex.get_path_from_root(sec_idx)
                title_path_obj = [node.meta_info.content for node in sec_path]
                sec_info = {
                    "id": sec_idx,
                    "title": section_title,
                    "path": title_path_obj,
                }
                if sec_entity_map and sec_idx in sec_entity_map:
                    entities_str = ", ".join(sec_entity_map[sec_idx])
                    sec_info["contained_entities"] = entities_str
                node_infos.append(sec_info)

            sec_info_str = json.dumps(node_infos, indent=2, ensure_ascii=False)
            return sec_info_str

        link_sec_str = (
            prep_nodes_json(link_nodes, sec_entity_map=sec_entity_map)
            if link_nodes
            else "[]"
        )
        remain_sec_str = (
            prep_nodes_json(remain_nodes, sec_entity_map=None) if remain_nodes else "[]"
        )
        query_prompt = LLM_EXPANSION_SELECT_PROMPT.format(
            user_question=query,
            primary_candidates_json=link_sec_str,
            remaining_sections_json=remain_sec_str,
        )

        return query_prompt

    def llm_section_selection(
        self,
        query: str,
        tree_nodes: List[dict],
        iter_context: Optional[SubStep] = None,
    ) -> None:
        """
        Use LLM to select the most relevant section based on the query and Section info.
        """
        sec_entity_map = self.link_section(tree_nodes)
        link_section_ids = list(sec_entity_map.keys())

        all_sections = self.gbc_index.TreeIndex.get_nodes_at_depth(self.select_depth)
        link_secs = [sec for sec in all_sections if sec.index_id in link_section_ids]
        remain_secs = [
            sec for sec in all_sections if sec.index_id not in link_section_ids
        ]
        iter_context.linked_section_ids = link_section_ids

        if len(remain_secs) == 0:
            log.info("No remaining sections to select from. Skipping LLM expansion.")
            iter_context.supplementary_ids = []
            iter_context.selected_explanation = (
                "No remaining sections for supplementary selection."
            )
            iter_context.retrieval_sec_ids = link_section_ids
            return

        query_prompt = self.prep_SecSel_prompt(
            query=query,
            link_nodes=link_secs,
            remain_nodes=remain_secs,
            sec_entity_map=sec_entity_map,
        )
        sel_ids = []
        explanation = "Error or no valid response from LLM during section expansion."

        remain_sec_ids_set = {sec.index_id for sec in remain_secs}
        try:
            res: SecEXPSelection = self.llm.get_json_completion(
                query_prompt, SecEXPSelection
            )
            if res:
                explanation = res.explanation
                if res.supplementary_ids:
                    # Validate the IDs returned by the LLM
                    for sup_id in res.supplementary_ids:
                        if sup_id in remain_sec_ids_set:
                            sel_ids.append(sup_id)
                        else:
                            log.warning(
                                f"LLM returned a supplementary ID {sup_id} which is not in the valid list of remaining sections. Ignoring it."
                            )

                    if sel_ids:
                        log.info(f"LLM selected {len(sel_ids)} supplementary sections.")
                    else:
                        log.info("LLM did not select any valid supplementary sections.")
                else:
                    log.info("LLM did not select any supplementary sections.")

        except Exception as e:
            log.error(f"Error occurred during section selection: {e}")

        iter_context.supplementary_ids = sel_ids
        iter_context.selected_explanation = explanation

        retrieval_sec_ids = list(set(link_section_ids + sel_ids))
        iter_context.retrieval_sec_ids = retrieval_sec_ids
        log.info(
            f"LLM selected {len(sel_ids)} supplementary sections, total {len(retrieval_sec_ids)} sections for retrieval."
        )

    def _process_retrieved_nodes(
        self, tree_data: List[Dict[str, Any]], iter_context: SubStep
    ) -> None:
        """Processes and categorizes retrieved nodes into the iteration context."""
        iter_context.retrieval_nodes = tree_data

        image_nodes = [node for node in tree_data if node["type"] == NodeType.IMAGE]
        text_nodes = [node for node in tree_data if node["type"] != NodeType.IMAGE]

        iter_context.iteration_image_nodes = image_nodes
        iter_context.iteration_text_nodes = text_nodes

    def get_GBC_info(self, iter_context: SubStep) -> None:
        """
        1. Get subgraph: sel_sec_id --> subtree --> subgraph.
        2. Use Three Layer Reranker to select most relevant TreeNodes in the subtree.
            2.1 PPR to rank the most relevant TreeNodes in the subtree.
            2.2 Rerank with text reranker model.
            2.3 Rerank with Multimodal method.
            Then: Skyline algorithm to select the most relevant TreeNodes.
        3. Combine the connected TreeNodes and Subgraph info to form the final GBC data info.
        """

        # 1. Get subgraph: sel_sec_id --> subtree --> subgraph.
        # Get the subtree rooted at the selected section ID
        if self.varient == "wo_selector":
            log.info("Variant 'wo_selector' selected")
            subtree_nodes = self.gbc_index.TreeIndex.get_nodes(hasRoot=False)
        else:
            log.info(f"Using {self.varient} variant for retrieval.")
            retrieval_sec_ids = iter_context.retrieval_sec_ids
            subtree_nodes = self.gbc_index.TreeIndex.get_subtree_nodes(retrieval_sec_ids)

        subtree_ids = [node.index_id for node in subtree_nodes]

        subgraph: nx.Graph = self.gbc_index.GraphIndex.get_kg_subgraph(subtree_ids)

        start_ent_map = iter_context.gbc_entity_map

        tree_node_ids, res_entities = self.retriever.skyline_filter(
            iter_context.sub_query, subtree_nodes, subgraph, start_ent_map
        )

        log.info(f"After skyline filtering, select {len(tree_node_ids)} TreeNodes")

        Graph_data = self.gbc_index.GraphIndex.get_subgraph_data(res_entities)
        iter_context.iteration_graph_nodes = Graph_data.get("nodes", [])

        tree_data = self.gbc_index.TreeIndex.get_nodes_data(tree_node_ids)
        self._process_retrieved_nodes(tree_data, iter_context)

    def _retrieve(
        self,
        query: str,
        iter_context: SubStep = None,
    ) -> None:
        """
        GBC retrieval following the steps:
        1. Extract entities from the query.
        2. Get the section nodes based on the entities.
        3. Use LLM to select the most relevant section based on the query and Section info.
        4. Use graph-based retrieval on the subgraph projected by the subtree (Select Section).

        iter_context: IterationStep, Iteration context for the current step.
        """

        Qent_GBCent_map = self._get_query_entity(query)
        iter_context.gbc_entity_map = Qent_GBCent_map

        tree_nodes = self.link_tree_node(Qent_GBCent_map)
        iter_context.linked_tree_nodes = tree_nodes

        # 3. Use LLM to select the most relevant section or supplementary sections
        if self.varient == "wo_selector":
            log.info("Variant 'wo_selector' selected: Skipping LLM section selection.")
            iter_context.retrieval_sec_ids = [self.gbc_index.TreeIndex.root_node.index_id]
        else:
            self.llm_section_selection(query, tree_nodes, iter_context)

        # 4. Graph-based retrieval on subgraph projected by the subtree (Select Section)
        self.get_GBC_info(iter_context)

    def process_analysis(self, context: GBCRAGContext, query_analysis: PlanResult):
        log.info(f"Query analysis type: {query_analysis.query_type}")

        if query_analysis.query_type == "simple":
            query = query_analysis.original_query
            current_step = SubStep(sub_query=query, sub_number=1)
            self._retrieve(query, current_step)

            final_answer, partial_answers = self.answer.answer_simple_question(
                query=query,
                retrieved_nodes=current_step.retrieval_nodes,
                entities=current_step.iteration_graph_nodes,
            )
            current_step.partial_answers = partial_answers
            current_step.generated_answer = final_answer

            context.iterations.append(current_step)
            context.final_answer = final_answer
        elif query_analysis.query_type == "complex":
            # 1. Separate retrieval tasks from the full plan
            retrieval_tasks = [
                sub_q
                for sub_q in query_analysis.sub_questions
                if sub_q.type == "retrieval"
            ]

            # 2. Execute each retrieval task and collect the results
            sub_question_results = []
            for i, task in enumerate(retrieval_tasks):
                sub_question = task.question
                current_step = SubStep(sub_query=sub_question, sub_number=i + 1)
                self._retrieve(sub_question, current_step)

                sub_answer, partial_answers = self.answer.answer_simple_question(
                    query=sub_question,
                    retrieved_nodes=current_step.retrieval_nodes,
                    entities=current_step.iteration_graph_nodes,
                )
                current_step.partial_answers = partial_answers
                current_step.generated_answer = sub_answer
                context.iterations.append(current_step)

                sub_question_results.append(
                    {"question": sub_question, "answer": sub_answer}
                )
            final_answer = self.answer.answer_complex_question(
                original_query=query_analysis.original_query,
                sub_question_plan=query_analysis.sub_questions,  # Pass the full plan
                sub_question_results=sub_question_results,  # Pass the results of the retrieval steps
            )
            context.final_answer = final_answer

        elif query_analysis.query_type == "global":
            # Create a step for the global operation
            current_step = SubStep(
                sub_query=query_analysis.original_query, sub_number=1
            )

            # 1. Filter the tree nodes based on the plan's filters
            filtered_nodes: List[TreeNode] = filter_tree_nodes(
                self.gbc_index.TreeIndex, query_analysis.filters
            )
            current_step.retrieval_nodes = filtered_nodes

            filter_nodes_ids = [node.index_id for node in filtered_nodes]
            tree_data = self.gbc_index.TreeIndex.get_nodes_data(filter_nodes_ids)
            self._process_retrieved_nodes(tree_data, current_step)
            log.info(f"Global filter resulted in {len(filtered_nodes)} nodes.")

            operation = query_analysis.operation.upper()

            # 2. Perform the specified operation
            if operation == "COUNT":
                # Direct calculation, no LLM call needed for the final step
                count_result = len(filtered_nodes)
                # You can format this into a more natural sentence if desired
                final_answer = (
                    f"Based on my analysis of the document, I found {count_result} items"
                    f" that answer the question: '{query_analysis.original_query}'"
                )

                current_step.partial_answers = [
                    {"source": "Direct Count", "content": final_answer}
                ]
            else:  # For LIST, SUMMARIZE, ANALYZE
                # Call the dedicated global answer agent method
                final_answer, partials = self.answer.answer_global_question(
                    original_query=query_analysis.original_query,
                    operation=operation,
                    filtered_nodes=current_step.retrieval_nodes,
                )
                current_step.partial_answers = partials

            context.iterations.append(current_step)
            context.final_answer = final_answer
        else:
            log.warning(f"Unknown query type: {query_analysis.query_type}")
            context.final_answer = "I'm sorry, I cannot process this query."

    def _create_augmented_prompt(self, query: str) -> str:
        pass

    def generation(self, query: str, query_output_dir: str):
        context = GBCRAGContext(query=query)

        if self.varient == "wo_plan":
            log.info("Variant 'wo_plan' selected: Skipping LLM planning.")
            query_analysis = PlanResult(
                query_type="simple",
                original_query=query,
            )
        else:
            query_analysis: PlanResult = self.planner.analyze(query)

        context.plan = query_analysis
        self.process_analysis(context, query_analysis)

        log.info(f"Final answer for query '{query}': {context.final_answer}")
        retrieval_ids = self._save_retrieval_res(context, query_output_dir)

        return context.final_answer, retrieval_ids

    def _save_retrieval_res(self, context: GBCRAGContext, query_output_dir: str):
        retrieval_ids = []

        # direct save the context to a json file
        retrieval_save_res = query_output_dir / "retrieval_res.json"
        context_dict = context.model_dump()
        with open(retrieval_save_res, "w", encoding="utf-8") as f:
            json.dump(context_dict, f, indent=2, ensure_ascii=False)
        log.info(f"Retrieval results saved to {retrieval_save_res}")

        # use the tree nodes as retrieval ids
        retrieval_ids = []
        for iter_step in context.iterations:
            text_nodes = iter_step.iteration_text_nodes
            if text_nodes:
                for node in text_nodes:
                    node_id = node.get("index_id")
                    if node_id is not None and node_id not in retrieval_ids:
                        retrieval_ids.append(node_id)
            image_nodes = iter_step.iteration_image_nodes
            if image_nodes:
                for node in image_nodes:
                    node_id = node.get("index_id")
                    if node_id is not None and node_id not in retrieval_ids:
                        retrieval_ids.append(node_id)

        retrieval_ids = sorted(retrieval_ids)

        return retrieval_ids

    def close(self):
        self.embedder.close()
        self.reranker.close()
        # if hasattr(self, 'mm_reranker'):
        #     self.mm_reranker.close()
        return super().close()
