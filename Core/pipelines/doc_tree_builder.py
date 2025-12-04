from Core.Index.Tree import DocumentTree
from Core.pipelines.tree_node_builder import create_node_by_type
from Core.pipelines.outline_extractor import extract_pdf_outline_in_chunks
from Core.pipelines.pdf_refiner import pdf_info_refiner
from Core.provider.extract_pdf_info import parse_doc, merge_middle_content
from Core.pipelines.tree_node_summary import generate_tree_node_summary
from Core.configs.system_config import SystemConfig
from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.provider.TokenTracker import TokenTracker
import os
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def construct_tree_index(
    tree_index: DocumentTree, pdf_list: list[dict], title_outline: list[dict]
) -> DocumentTree:
    """Constructs the tree index from the provided PDF content and title outline.
    :param tree_index: DocumentTree instance to construct the index.
    :param pdf_list: List of dictionaries containing PDF content.
    :param title_outline: List of dictionaries containing title outline information.
    :return: The updated DocumentTree instance with the constructed index.
    """

    for content in title_outline:
        node = create_node_by_type(pdf_content=content, isTitle=True)
        tree_index.add_node(node)

        # Add parent node by parent_id
        text_level = content.get("text_level", -1)
        if text_level == 0:
            # If text_level is 0, it is a root node
            tree_index.root_node.add_child(node)
        else:
            parent_id = content.get("parent_id", None)
            if parent_id is not None:
                parent_node = tree_index.get_node_by_pdf_id(parent_id)
                if parent_node:
                    parent_node.add_child(node)
            else:
                # If no parent_id, add to root
                tree_index.root_node.add_child(node)

        # Add child nodes
        end_idx = content["end_id"]
        for i in range(content["pdf_id"], end_idx):
            if i == len(pdf_list):
                break  # Avoid index out of range
            child_i = pdf_list[i]
            content_id = child_i.get("pdf_id", -1)
            if content_id > content["pdf_id"] and content_id < end_idx:
                child_node = create_node_by_type(pdf_content=child_i, isTitle=False)
                tree_index.add_node(child_node)
                node.add_child(child_node)

    log.info(f"Total {len(tree_index.nodes)} nodes added to the tree index.")
    return tree_index


def build_tree_from_pdf(cfg: SystemConfig, reforce: bool = False) -> DocumentTree:

    tree_index_path = DocumentTree.get_save_path(cfg.save_path)
    if os.path.exists(tree_index_path) and not reforce:
        # Load existing tree index
        log.info(f"Loading existing tree index from {tree_index_path}...")
        tree_index = DocumentTree.load_from_file(tree_index_path)
        log.info("Tree index loaded successfully.")
        return tree_index
    else:
        # Create a new tree index
        log.info("Creating a new tree index...")

    meta_dict = {
        "file_name": os.path.basename(cfg.pdf_path),
        "file_path": cfg.pdf_path,
    }

    os.makedirs(cfg.save_path, exist_ok=True)

    tree_index = DocumentTree(meta_dict=meta_dict, cfg=cfg)

    backend = cfg.mineru.backend
    server_url = cfg.mineru.server_url
    method = cfg.mineru.method
    base_file_name = Path(cfg.pdf_path).stem
    tmp_save_path = os.path.join(
        cfg.save_path, method, f"{base_file_name}_merged_content.json"
    )

    if os.path.exists(tmp_save_path) and not reforce:
        # tmp load pdf_list
        import json

        with open(tmp_save_path, "rb") as f:
            pdf_list = json.load(f)
        print(f"Loaded content from {tmp_save_path}")
    else:
        # Extract content from the PDF file
        log.info(f"Extracting content from {cfg.pdf_path}...")
        middle_json, content_list = parse_doc(
            cfg.pdf_path,
            output_dir=cfg.save_path,
            backend=backend,
            method=method,
            server_url=server_url,
            lang=cfg.mineru.lang,
        )

        file_name = str(Path(cfg.pdf_path).stem)
        save_dir = os.path.join(cfg.save_path, method)
        pdf_list = merge_middle_content(
            middle_json,
            content_list,
            parse_dir=os.path.join(cfg.save_path, method),
            save_dir=save_dir,
            file_name=file_name,
        )  # merge middle json content with content list.

        # tmp pdf_list save for fast test
        log.info(f"Content extracted and saved to {tmp_save_path}")

    llm = LLM(cfg.llm)
    vlm = VLM(cfg.vlm) if cfg.tree.use_vlm else None

    pdf_list = pdf_info_refiner(pdf_list, llm)
    title_outline = extract_pdf_outline_in_chunks(pdf_list, llm)
    tree_index = construct_tree_index(
        tree_index=tree_index, pdf_list=pdf_list, title_outline=title_outline
    )
    token_tracker = TokenTracker.get_instance()
    tree_index_cost = token_tracker.record_stage("tree_index_construction")
    log.info(f"Tree index construction cost: {tree_index_cost}")

    if cfg.tree.node_summary:
        # Generate summaries for each node
        tree_index = generate_tree_node_summary(
            tree_index=tree_index,
            llm=llm,
            use_VLM=cfg.tree.use_vlm,
            vlm=vlm,
        )
        token_tracker = TokenTracker.get_instance()
        summary_cost = token_tracker.record_stage("tree_node_summary")
        log.info(f"Tree node summary generation cost: {summary_cost}")

    # save
    tree_index.save_to_file()
    return tree_index
