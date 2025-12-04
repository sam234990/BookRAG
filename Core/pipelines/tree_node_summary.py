from typing import Optional, List, Dict

from Core.Index.Tree import TreeNode, NodeType, DocumentTree
from Core.prompts.summary_prompt import NODE_SUMMARY_PROMPT, SEC_SUMMARY_PROMPT
from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.utils.utils import num_tokens, TextProcessor
import os
import logging

log = logging.getLogger(__name__)


def get_node_summary_prompt(tree_node: TreeNode, max_token: int) -> str:
    node_type = tree_node.type
    if node_type not in [
        NodeType.TEXT,
        NodeType.IMAGE,
        NodeType.TABLE,
        NodeType.EQUATION,
    ]:
        log.warning(f"Node type {node_type} is not supported for summary generation.")
        return ""

    if node_type in [NodeType.IMAGE, NodeType.TABLE]:
        content = (
            "This is an image." if node_type == NodeType.IMAGE else "This is a table."
        )
        content += "Here is the caption: "
        content += tree_node.meta_info.content or ""
        content += (
            f"\n{tree_node.meta_info.table_body}" if node_type == NodeType.TABLE else ""
        )
    else:
        content = tree_node.meta_info.content or ""

    # 2. 【新增】检查并截断内容
    base_prompt_tokens = num_tokens(NODE_SUMMARY_PROMPT.format(node_text=""))
    available_tokens = max_token - base_prompt_tokens
    if num_tokens(content) > available_tokens:
        log.warning(
            f"Content length ({num_tokens(content)} tokens) exceeds max_token ({max_token}). Truncating."
        )
        # 调用静态方法进行切分
        chunks = TextProcessor.split_text_into_chunks(text=content, max_length=max_token)
        # 只取第一个分片
        content = chunks[0] if chunks else ""

    prompt = NODE_SUMMARY_PROMPT.format(node_text=content)
    return prompt


def generate_node_summary(
    tree_node: TreeNode, llm: LLM, use_VLM: bool = False, vlm: Optional[VLM] = None
) -> str:
    """Generate a summary for a single tree node.
    This function uses the LLM to generate a summary based on the node's content.
    If the node is an image or table, it will use the VLM if provided.
    """
    node_type = tree_node.type
    prompt = get_node_summary_prompt(tree_node, max_token=llm.config.max_tokens)

    if use_VLM and vlm is not None and node_type in [NodeType.IMAGE, NodeType.TABLE]:
        # Use VLM for image or table nodes
        image_path = tree_node.meta_info.img_path
        if not os.path.exists(image_path):
            log.warning(
                f"Image path {image_path} does not exist for node {tree_node.index_id}."
            )
            return ""
        summary = vlm.generate(prompt_or_memory=prompt, images=[image_path])
    else:
        summary = llm.get_completion(prompt=prompt, json_response=False)

    if not summary:
        log.warning(f"Failed to generate summary for node {tree_node.index_id}.")
        return ""
    log.info(f"Generated summary for node {tree_node.index_id}: {summary}")
    return summary.strip()


def get_sec_summary_prompt(sec_node: TreeNode, max_token: int) -> str:
    """Get the prompt for generating a section summary.
    This function formats the section text and its immediate children's summaries into a prompt.
    """
    base_prompt_tokens = num_tokens(
        SEC_SUMMARY_PROMPT.format(section_text="", content_summary="")
    )
    available_tokens = max_token - base_prompt_tokens

    if available_tokens <= 0:
        log.warning(
            f"max_token ({max_token}) is too small to even fit the prompt template."
        )
        return ""

    # Get initial content
    section_text = sec_node.meta_info.content or ""

    def get_children_text(node: TreeNode) -> str:
        """Recursively get text from all immediate children nodes."""
        if not node.children:
            return ""
        children_text = []
        child_prefix = {
            NodeType.TEXT: "Text: ",
            NodeType.IMAGE: "Image: ",
            NodeType.TABLE: "Table: ",
            NodeType.EQUATION: "Equation: ",
        }
        for child in node.children:
            child_text = child.summary or child.meta_info.content or ""
            child_type = child.type
            if child_text:
                children_text.append(f"{child_prefix.get(child_type, '')}{child_text}")
        return "\n".join(children_text)

    children_text = get_children_text(sec_node)

    # Truncation Logic
    section_tokens = num_tokens(section_text)

    if section_tokens >= available_tokens:
        # If section_text alone is too long, truncate it and use no children_text
        log.warning(
            f"Section text ({section_tokens} tokens) exceeds available tokens ({available_tokens}). Truncating section text and omitting children summaries."
        )
        chunks = TextProcessor.split_text_into_chunks(
            text=section_text, max_length=available_tokens
        )
        section_text = chunks[0] if chunks else ""
        children_text = ""
    else:
        # If section_text fits, see how much of children_text can be included
        remaining_tokens = available_tokens - section_tokens
        children_tokens = num_tokens(children_text)
        if children_tokens > remaining_tokens:
            log.warning(
                f"Children summaries ({children_tokens} tokens) exceed remaining tokens ({remaining_tokens}). Truncating children summaries."
            )
            chunks = TextProcessor.split_text_into_chunks(
                text=children_text, max_length=remaining_tokens
            )
            children_text = chunks[0] if chunks else ""

    return SEC_SUMMARY_PROMPT.format(
        section_text=section_text, content_summary=children_text
    )


def generate_section_summary(sec_node: TreeNode, llm: LLM) -> str:
    """Generate a summary for a section node.
    This function uses the LLM to generate a summary based on the section's content and its children.
    It includes text from the section itself and summaries of its immediate children.
    """
    prompt = get_sec_summary_prompt(sec_node, llm.config.max_tokens)
    summary = llm.get_completion(prompt=prompt, json_response=False)
    return summary


def generate_tree_node_summary(
    tree_index: DocumentTree, llm: LLM, use_VLM: bool = False, vlm: Optional[VLM] = None
) -> DocumentTree:
    """Generate summaries for each node in the tree index.
    The generating order is from the leaf nodes to the root node.
    """
    log.info("Generating summaries for tree nodes...")

    def get_nodes_by_level_bottom_up(node, current_level=0, level_dict=None):
        """Get all nodes organized by level from bottom to top"""
        if level_dict is None:
            level_dict = {}

        if current_level not in level_dict:
            level_dict[current_level] = []

        level_dict[current_level].append(node)

        # Recursively process children
        for child in node.children:
            get_nodes_by_level_bottom_up(child, current_level + 1, level_dict)

        return level_dict

    # Get all nodes organized by level from bottom to top
    level_dict = get_nodes_by_level_bottom_up(tree_index.root_node)
    log.info(f"Processing tree with {len(level_dict)} levels")

    # Process nodes from the bottom level to the top
    for level in sorted(level_dict.keys(), reverse=True):
        log.info(f"Processing level {level} with {len(level_dict[level])} nodes.")
        # Initialize lists for LLM and VLM prompts
        llm_prompt_list = []
        llm_node_idx_list = []

        vlm_prompt_list = []
        vlm_images_list = []
        vlm_node_idx_list = []

        for node in level_dict[level]:
            if node == tree_index.root_node:
                # we skip the root node
                continue
            children_len = len(node.children)
            if children_len == 0:
                # Leaf node, generate summary directly
                summary_prompt = get_node_summary_prompt(
                    node, max_token=llm.config.max_tokens
                )
                if use_VLM and node.type in [NodeType.IMAGE, NodeType.TABLE]:
                    # Use VLM for image or table nodes
                    image_path = node.meta_info.img_path
                    if not os.path.exists(image_path):
                        log.warning(
                            f"Image path {image_path} does not exist for node {node.index_id}."
                        )
                        continue
                    vlm_prompt_list.append(summary_prompt)
                    vlm_images_list.append(image_path)
                    vlm_node_idx_list.append(node.index_id)
                else:
                    # Use LLM for text nodes or if VLM is not used
                    llm_prompt_list.append(summary_prompt)
                    llm_node_idx_list.append(node.index_id)
            else:
                # Non-leaf node, prepare for section summary
                summary_prompt = get_sec_summary_prompt(node, llm.config.max_tokens)
                llm_prompt_list.append(summary_prompt)
                llm_node_idx_list.append(node.index_id)
        # Generate summaries using LLM
        if llm_prompt_list:
            log.info(
                f"Generating summaries for {len(llm_prompt_list)} nodes using LLM."
            )
            llm_summaries = llm.batch_get_completion(
                prompts=llm_prompt_list, json_response=False
            )
            for idx, summary in zip(llm_node_idx_list, llm_summaries):
                node = tree_index.get_node_by_index_id(idx)
                if node:
                    node.summary = summary.strip()
                    log.info(f"Node {idx} summary generated: {summary.strip()}")
                else:
                    log.warning(f"Node with ID {idx} not found in the tree index.")

        # Generate summaries using VLM if applicable
        if use_VLM and vlm_prompt_list:
            log.info(
                f"Generating summaries for {len(vlm_prompt_list)} nodes using VLM."
            )
            vlm_summaries = vlm.batch_generate(
                query=vlm_prompt_list, images=vlm_images_list
            )
            for idx, summary in zip(vlm_node_idx_list, vlm_summaries):
                node = tree_index.get_node_by_index_id(idx)
                if node:
                    node.summary = summary.strip()
                    log.info(f"Node {idx} summary generated: {summary.strip()}")
                else:
                    log.warning(f"Node with ID {idx} not found in the tree index.")

    log.info("All node summaries generated successfully.")
    # Return the updated tree index with summaries

    return tree_index


if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        logging.basicConfig(
            level=logging.INFO,  # 或 logging.DEBUG
            format="%(asctime)s %(levelname)s %(message)s",
        )
    tmp_path = "/home/wangshu/multimodal/GBC-RAG/test/tree_index"
    tree_index = DocumentTree.load_from_file(DocumentTree.get_save_path(tmp_path))
    from Core.configs.system_config import load_system_config

    cfg = load_system_config("/home/wangshu/multimodal/GBC-RAG/config/default.yaml")

    llm = LLM(llm_config=cfg.llm)

    tree_index = generate_tree_node_summary(tree_index=tree_index, llm=llm)
    one_step_index_1 = tree_index.get_one_depth_summary(1)
    print(f"Node ID: 1, Summary: \n")
    print(one_step_index_1)
