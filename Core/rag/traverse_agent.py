from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.Index.Tree import DocumentTree, TreeNode, NodeType, MetaInfo
from Core.rag.base_rag import BaseRAG
from Core.prompts.traverseagent_prompt import (
    NAVIGATOR_PROMPT_TEMPLATE,
    ANSWER_GENERATOR_INSTRUCTION_TEMPLATE,
    NavigatorDecision,
)
from Core.configs.rag.traverse_config import TraverseRAGConfig

import json
import random
from typing import List, Any, Tuple, Optional
import logging


log = logging.getLogger(__name__)


class TraverseAgent(BaseRAG):
    def __init__(
        self,
        config: TraverseRAGConfig,
        llm: LLM,
        vlm: Optional[VLM] = None,
        tree_index: Optional[DocumentTree] = None,
    ):
        super().__init__(
            llm, name="Traverse Agent", description="Tree Traverse-based RAG Agent"
        )
        self.vlm = vlm
        self.tree_index = tree_index

        # Extract parameters from the config object
        self.max_depth = config.max_depth

        # You can store the whole config if needed for other parameters
        self.config = config

    def _create_navigator_prompt(
        self, query: str, current_node: TreeNode, child_nodes: List[TreeNode]
    ) -> str:
        """
        Generates a structured prompt for the LLM to decide which child node to explore next.
        This prompt includes the user's query, a summary of the current node, and a JSON array
        of available child nodes with their summaries and relevant metadata.
        """
        options_list = []
        for i, child in enumerate(child_nodes, 1):
            if not child.summary:
                continue

            meta = child.meta_info
            option_data = {
                "choice_number": i,
                "type": child.type.upper(),
                "summary": child.summary,
            }

            if child.type in [NodeType.TITLE, NodeType.EQUATION] and meta.content:
                option_data["content"] = meta.content
            elif child.type in [NodeType.TABLE, NodeType.IMAGE] and meta.caption:
                option_data["caption"] = meta.caption
            elif child.type == NodeType.TEXT and meta.content:
                words = meta.content.split()
                preview_words = words[:50]
                preview_text = " ".join(preview_words)
                if len(words) > 50:
                    preview_text += "..."
                if preview_text:
                    option_data["content_preview"] = preview_text

            options_list.append(option_data)

        if options_list:
            options_str = json.dumps(options_list, indent=2)
        else:
            options_str = "No further nodes available."

        current_summary = current_node.summary or "This is the root of the document."

        return NAVIGATOR_PROMPT_TEMPLATE.format(
            query=query, current_summary=current_summary, options_str=options_str
        )

    def _retrieve(self, query: str) -> List[TreeNode]:
        """
        Performs intelligent traversal using structured JSON calls to the LLM.
        """
        if not self.tree_index or not self.tree_index.root_node:
            return []

        # set the maximum depth for traversal
        max_depth = self.tree_index.get_max_depth() + 1  # +1 for root node
        if self.max_depth != -1:
            # set -1 to traversal all
            max_depth = min(max_depth, self.max_depth)
        

        current_node = self.tree_index.root_node
        traversal_path: List[TreeNode] = []

        for i in range(max_depth):
            traversal_path.append(current_node)
            child_nodes = current_node.children

            if not child_nodes:
                log.info(
                    f"INFO: No child nodes found for current node (ID: {current_node.index_id}). Stopping traversal."
                )
                break

            if len(child_nodes) == 1:
                log.info(
                    f"INFO: Only one child node found (ID: {child_nodes[0].index_id}). Automatically selecting this node."
                )
                current_node = child_nodes[0]
                continue

            try:
                # Create the decision-making prompt
                decision_prompt = self._create_navigator_prompt(
                    query, current_node, child_nodes
                )

                # Use get_json_completion for robust, structured output
                decision_obj = self.llm.get_json_completion(
                    prompt=decision_prompt, schema=NavigatorDecision
                )

                if not decision_obj or not isinstance(decision_obj, NavigatorDecision):
                    raise ValueError("LLM returned an invalid or null decision object.")

                choice = decision_obj.choice
                reason = decision_obj.reason
                log.info(
                    f"INFO: At Depth {i+1}, Node {current_node.index_id or 'root'}: LLM chose option {choice}. Reason: '{reason}'"
                )

                if choice == 0 or not (1 <= choice <= len(child_nodes)):
                    log.info(
                        "INFO: LLM decided to stop or made an invalid choice. Halting traversal."
                    )
                    break

                current_node = child_nodes[choice - 1]

            except Exception as e:
                log.error(
                    f"ERROR: An error occurred during LLM navigation: {e}. "
                    f"Activating fallback: randomly selecting a child node."
                )
                # Fallback: randomly select one of the child nodes
                current_node = random.choice(child_nodes)
                log.info(
                    f"INFO: Fallback activated. Randomly selected child node with ID: {current_node.index_id}"
                )

        return traversal_path

    def _create_augmented_prompt(
        self, query: str, context_nodes: List[TreeNode]
    ) -> Tuple[str, List[str]]:
        """
        Constructs the final prompt for answer generation, including all relevant context.
        """
        context_str_parts = []
        image_paths = []

        if not context_nodes:
            context_str_parts.append(
                "No relevant information was found in the document."
            )
        else:
            for node in context_nodes:
                node_type = node.type
                meta = node.meta_info

                context_str_parts.append(f"\n## Context (Type: {node_type})")

                if (
                    node_type in [NodeType.TEXT, NodeType.TITLE, NodeType.EQUATION]
                    and meta.content
                ):
                    context_str_parts.append(meta.content)
                elif node_type == NodeType.TABLE and meta.table_body:
                    table_context_parts = []
                    table_context_parts.append(f"Table Content: {meta.content}")
                    table_context_parts.append(f"Table Body:\n{meta.table_body}")
                    if table_context_parts:
                        context_str_parts.append("\n\n".join(table_context_parts))

                elif node_type == NodeType.IMAGE and meta.img_path:
                    image_paths.append(meta.img_path)
                    image_context_parts = []
                    image_context_parts.append(f"Image Content: {meta.content}")
                    if image_context_parts:
                        context_str_parts.append("\n".join(image_context_parts))

        final_context_str = "\n".join(context_str_parts)

        final_prompt = ANSWER_GENERATOR_INSTRUCTION_TEMPLATE.format(
            query=query, context_str=final_context_str
        )

        return final_prompt, image_paths

    def _save_retrieval_res(
        self, context_nodes: List[TreeNode], query_output_dir: str
    ) -> List[Any]:
        """
        Saves the retrieval results (context nodes) to the specified output directory.
        """
        retrieval_node_ids = []
        for node in context_nodes:
            node_id = node.index_id
            meta_info_dict = {
                "id": node_id,
                "type": node.type,
                "content": node.meta_info.content,
                "summary": node.summary,
                "img_path": node.meta_info.img_path,
            }
            retrieval_node_ids.append(node_id)
            node_file_path = query_output_dir / f"{node_id}.json"
            with open(node_file_path, "w", encoding="utf-8") as f:
                json.dump(meta_info_dict, f, indent=2, ensure_ascii=False)
        log.info(f"final save the retrieval res into {query_output_dir}")
        return retrieval_node_ids

    def generation(
        self, query: str, query_output_dir: str
    ) -> Tuple[str, List[Any]]:  # 实际实现
        """
        Executes the full RAG flow.
        (This method's logic remains unchanged)
        """
        context_nodes = self._retrieve(query)

        # 1. Get the augmented prompt and image paths
        final_prompt, image_paths = self._create_augmented_prompt(query, context_nodes)

        if image_paths and self.vlm:
            log.info(
                f"INFO: Image context found ({len(image_paths)} images). Using VLM for generation."
            )
            final_answer = self.vlm.generate(prompt_or_memory=final_prompt, images=image_paths)
        else:
            log.info("INFO: Text-only context. Using LLM for generation.")
            final_answer = self.llm.get_completion(prompt=final_prompt)

        # 2. Convert context nodes to a structured list for output
        retrieval_node_ids = self._save_retrieval_res(context_nodes, query_output_dir)
        return final_answer, retrieval_node_ids

    def close(self):
        return super().close()
