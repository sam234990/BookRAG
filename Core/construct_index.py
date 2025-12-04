import os
import logging
import time
import pandas as pd

from Core.configs.vdb_config import VDBConfig

log = logging.getLogger(__name__)

from Core.Index.GBCIndex import GBC
from Core.configs.system_config import SystemConfig
from Core.pipelines.doc_tree_builder import build_tree_from_pdf
from Core.pipelines.kg_builder import build_knowledge_graph
from Core.pipelines.vdb_index import (
    build_other_vdb_index,
    build_vdb_index,
    compute_mm_embedding,
    compute_mm_embedding_question,
)
from Core.provider.TokenTracker import TokenTracker
from Core.utils.file_utils import save_indexing_stats


def construct_GBC_index(cfg: SystemConfig, tree_only: bool = False):
    """
    Construct the GBC index from the document tree and knowledge graph.

    :param cfg: Configuration object containing settings for the index construction.
    :return: A tuple containing the DocumentTree and Graph objects.
    """
    log.info("Starting GBC index construction...")

    token_tracker = TokenTracker.get_instance()
    token_tracker.reset()

    # This dictionary will hold all stats for the CURRENT run
    current_run_stats = {}

    # --- Measure Tree Building ---
    tree_start_time = time.time()
    tree_index = build_tree_from_pdf(cfg)
    tree_duration = time.time() - tree_start_time
    log.info(f"Document tree constructed in {tree_duration:.2f} seconds.")
    current_run_stats["build_tree_time"] = round(tree_duration, 2)

    if tree_only:
        log.info("Only build tree index. Finished.")
        # Add final token usage to our stats dictionary
        current_run_stats["token_stage_history"] = token_tracker.stage_history

        # Save all collected stats and exit
        save_indexing_stats(save_path=cfg.save_path, new_stats=current_run_stats)
        return tree_index

    # --- Measure Knowledge Graph Building ---
    kg_start_time = time.time()
    graph_index = build_knowledge_graph(tree_index, cfg)

    # The 'kg_extraction' stage is recorded inside build_knowledge_graph
    gbc_index = GBC(config=cfg, graph_index=graph_index, TreeIndex=tree_index)
    gbc_index.save_gbc_index()

    # rebuild vdb
    gbc_index.rebuild_vdb()

    kg_duration = time.time() - kg_start_time
    log.info(f"Knowledge graph constructed and saved in {kg_duration:.2f} seconds.")
    current_run_stats["build_kg_time"] = round(kg_duration, 2)

    # --- Finalize and Save All Stats for the Full Run ---
    log.info("Full GBC index construction finished. Saving final stats...")
    current_run_stats["token_stage_history"] = token_tracker.stage_history

    save_indexing_stats(save_path=cfg.save_path, new_stats=current_run_stats)

    return gbc_index

def rebuild_graph_vdb(cfg: SystemConfig):
    gbc_index = GBC.load_gbc_index(cfg)
    gbc_index.rebuild_vdb()
    log.info("Rebuilt graph VDB successfully.")


def construct_vdb(cfg: SystemConfig):
    token_tracker = TokenTracker.get_instance()
    token_tracker.reset()

    log.info("Starting vector database construction...")

    if cfg.index_type in ["vanilla", "bm25", "raptor"]:
        log.info(f"Index type is {cfg.index_type}. Start building other vdb index...")
        build_other_vdb_index(cfg)
        return

    current_run_stats = {}

    tree_start_time = time.time()
    tree_index = build_tree_from_pdf(cfg)
    tree_duration = time.time() - tree_start_time
    log.info(f"Document tree constructed in {tree_duration:.2f} seconds.")
    current_run_stats["build_tree_time"] = round(tree_duration, 2)

    log.info("Document tree constructed successfully for vector database.")

    current_run_stats["token_stage_history"] = token_tracker.stage_history

    # Save all collected stats and exit
    save_indexing_stats(save_path=cfg.save_path, new_stats=current_run_stats)

    vdb_cfg: VDBConfig = cfg.vdb
    if cfg.save_path not in vdb_cfg.vdb_dir_name:
        vdb_cfg.vdb_dir_name = os.path.join(cfg.save_path, vdb_cfg.vdb_dir_name)
    log.info(f"Vector database path set to: {vdb_cfg.vdb_dir_name}")

    # if exist the dir, remove and rebuild vdb
    if os.path.exists(vdb_cfg.vdb_dir_name) and not vdb_cfg.force_rebuild:
        log.info(f"Vector database path already exists: {vdb_cfg.vdb_dir_name}. Skip")
        return

    if vdb_cfg.force_rebuild and os.path.exists(vdb_cfg.vdb_dir_name):
        log.info(
            f"Vector database path already exists: {vdb_cfg.vdb_dir_name}. Remove and rebuild"
        )
        import shutil

        shutil.rmtree(vdb_cfg.vdb_dir_name)

    os.makedirs(os.path.dirname(vdb_cfg.vdb_dir_name), exist_ok=True)

    vbd_start_time = time.time()
    build_vdb_index(tree_index, vdb_cfg)
    vdb_duration = time.time() - vbd_start_time
    log.info(f"Vector database constructed in {vdb_duration:.2f} seconds.")

    current_run_stats["build_vdb_time"] = round(vdb_duration, 2)

    # Save all collected stats and exit
    save_indexing_stats(save_path=cfg.save_path, new_stats=current_run_stats)


def compute_mm_reranker(cfg: SystemConfig, group: pd.DataFrame):

    tree_index = build_tree_from_pdf(cfg)

    compute_mm_embedding(cfg, tree_index)
    
    compute_mm_embedding_question(cfg, group)


if __name__ == "__main__":
    print("test")

    # parser = argparse.ArgumentParser(description="Extract text content from PDF files.")
    # parser.add_argument(
    #     "--config_path",
    #     type=str,
    #     default="/home/wangshu/multimodal/GBC-RAG/config/gbc.yaml",
    #     help="Path to the configuration file.",
    # )

    # args = parser.parse_args()

    # cfg = load_system_config(args.config_path)

    # if not os.path.exists(cfg.save_path):
    #     os.makedirs(cfg.save_path)
    #     log.info(f"Created directory: {cfg.save_path}")
    # else:
    #     log.info(f"Directory already exists: {cfg.save_path}")

    # construct_vdb(cfg)

    # gbc_index = construct_GBC_index(cfg)
    # log.info("GBC index construction completed successfully.")
