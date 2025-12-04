import os
import pandas as pd
from Core.configs.system_config import load_system_config, SystemConfig
from Core.provider.TokenTracker import TokenTracker
from Core.rag import create_rag_agent
from Core.rag.base_rag import BaseRAG
from Core.utils.resource_loader import prepare_rag_dependencies

import json
from tqdm import tqdm
from pathlib import Path
import logging
import argparse
import time
from rich.logging import RichHandler

log = logging.getLogger(__name__)
# logging.basicConfig(
#     level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
# )


def run_rag(
    rag_agent: BaseRAG,
    output_dir: str,
    force_reprocess: bool = False,
    dataset_path: str = None,
    data_df: pd.DataFrame = None,
):
    log.info(f"Results will be saved to: {output_dir}")

    # load dataset
    dataset = None
    if dataset_path and os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    elif data_df is not None:
        # transform data_df into list of dict
        dataset = data_df.to_dict(orient="records")
    else:
        log.error(f"Dataset file not found: {dataset_path}")
        log.error("Dataframe data not provided")
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    results_list = []
    start_time = time.time()
    load_cnt = 0
    for i, item in enumerate(tqdm(dataset, desc=f"Processing Query")):
        query_index_str = f"query_{i+1:03d}"
        query_output_dir = output_dir / query_index_str
        query_result_file = query_output_dir / "result.json"

        if query_result_file.exists() and not force_reprocess:
            try:
                with open(query_result_file, "r", encoding="utf-8") as f:
                    existing_result = json.load(f)
                if existing_result.get("output"):
                    log.info(f"Skipping {query_index_str}, result already exists.")
                    results_list.append(existing_result)
                    load_cnt += 1
                    continue
            except (json.JSONDecodeError, KeyError):
                log.warning(
                    f"Found corrupted result file for {query_index_str}. Re-processing."
                )

        query = item.get("question")
        if not query:
            log.warning(f"Skipping item {i} due to missing 'question' field.")
            continue

        query_output_dir.mkdir(exist_ok=True)
        answer, retrieved_node_ids = rag_agent.generation(query, query_output_dir)

        current_result = {
            **item,
            "output": answer,
            "retrieved_node_ids": retrieved_node_ids,
        }
        with open(query_result_file, "w", encoding="utf-8") as f:
            json.dump(current_result, f, indent=2, ensure_ascii=False)

        results_list.append(current_result)

    end_time = time.time()
    total_time = end_time - start_time
    log.info(f"✅ RAG processing complete in {total_time:.2f} seconds.")
    final_res_path = output_dir / "final_results.json"
    with open(final_res_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)

    log.info(f"✅ RAG complete. All results are saved to {final_res_path}")
    rag_agent.close()

    token_tracker = TokenTracker.get_instance()
    rag_cost = token_tracker.record_stage("rag_cost")
    log.info(f"The token cost of RAG in the current document: {rag_cost}")

    update_and_save_cost(
        output_dir=output_dir,
        new_cost=rag_cost,
        new_time=total_time,
        load_cnt=load_cnt,
        dataset_len=len(dataset),
        force_reprocess=force_reprocess,
    )


def update_and_save_cost(
    output_dir: Path,
    new_cost: int,
    new_time: float,
    load_cnt: int,
    dataset_len: int,
    force_reprocess: bool,
):
    if load_cnt == dataset_len:
        log.info(f"All {load_cnt} samples were loaded from existing results.")
        log.info("Skipping saving token cost since no new inference was made.")
        return

    token_cost_path = output_dir / "token_cost.json"
    previous_cost = {}
    previous_time = 0

    if token_cost_path.exists() and load_cnt != 0 and not force_reprocess:
        log.info(
            f"Found existing cost file at {token_cost_path}. Reading previous values."
        )
        try:
            with open(token_cost_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            previous_cost = existing_data.get("rag_cost", {})
            previous_time = existing_data.get("time", 0)
            log.info(
                f"Previous cost: {previous_cost}, Previous time: {previous_time:.2f}s"
            )
        except (json.JSONDecodeError, KeyError):
            log.warning(
                f"Could not read or parse existing cost file. Starting from zero."
            )
            previous_cost = {}
            previous_time = 0

    total_rag_cost = previous_cost.copy()
    for key, value in new_cost.items():
        total_rag_cost[key] = total_rag_cost.get(key, 0) + value

    total_processing_time = previous_time + new_time

    final_token_cost = {
        "rag_cost": total_rag_cost,
        "time": total_processing_time,
    }

    log.info(
        f"Saving accumulated cost: {total_rag_cost}, Total time: {total_processing_time:.2f}s"
    )
    with open(token_cost_path, "w", encoding="utf-8") as f:
        json.dump(final_token_cost, f, indent=2, ensure_ascii=False)


def create_log_handler(cfg: SystemConfig, dataset_path: str):
    """
    Creates a logging handler that writes logs to a file in the specified output directory.
    The log file is named based on the dataset file name.
    Return: output_dir
    """
    rag_strategy = cfg.rag.strategy_config.strategy
    log.info(f"Using RAG strategy: {rag_strategy}")

    dataset_file = Path(dataset_path)
    output_dir = Path(cfg.save_path) / f"eval_{dataset_file.stem}_{rag_strategy}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = output_dir / "evaluation.log"

    # 给 root logger 添加 FileHandler
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            root_logger.removeHandler(h)
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(file_handler)
    root_logger.info(f"Logging to: {log_file_path}")

    return output_dir


def inference_base(cfg: SystemConfig, dataset_path: str):
    output_dir = create_log_handler(cfg, dataset_path)

    log.info(
        f"Successfully loaded config. Using RAG strategy: {cfg.rag.strategy_config.strategy}"
    )
    dependencies = prepare_rag_dependencies(cfg=cfg)

    rag_agent = create_rag_agent(
        strategy_config=cfg.rag.strategy_config,
        llm_config=cfg.llm,
        vlm_config=cfg.vlm,
        **dependencies,
    )
    log.info(f"RAG agent created with strategy: {rag_agent.name}")

    run_rag(
        rag_agent=rag_agent,
        dataset_path=dataset_path,
        output_dir=output_dir,
        force_reprocess=True,
    )


def inference(cfg: SystemConfig, data_df: pd.DataFrame, dataset_name: str):
    dependencies = prepare_rag_dependencies(cfg=cfg)
    rag_agent = create_rag_agent(
        strategy_config=cfg.rag.strategy_config,
        llm_config=cfg.llm,
        vlm_config=cfg.vlm,
        **dependencies,
    )
    log.info(f"RAG agent created with strategy: {rag_agent.name}")

    rag_strategy = cfg.rag.strategy_config.strategy
    log.info(f"Using RAG strategy: {rag_strategy}")
    if rag_strategy == "vanilla":
        retrieval_method = cfg.rag.strategy_config.retrieval_method
        output_dir = output_dir = (
            Path(cfg.save_path) / f"eval_{dataset_name}_{retrieval_method}"
        )
    elif rag_strategy == "gbc":
        varient = cfg.rag.strategy_config.varient
        output_dir = output_dir = (
            Path(cfg.save_path) / f"eval_{dataset_name}_{rag_strategy}_{varient}"
        )
    else:
        output_dir = output_dir = (
            Path(cfg.save_path) / f"eval_{dataset_name}_{rag_strategy}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rag(
        rag_agent=rag_agent,
        output_dir=output_dir,
        force_reprocess=cfg.rag_force_reprocess,
        data_df=data_df,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation on a dataset.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/wangshu/multimodal/GBC-RAG/config/gbc.yaml",
        # default="/home/wangshu/multimodal/GBC-RAG/config/mm.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the JSON dataset file with questions.",
        default="/home/wangshu/multimodal/GBC-RAG/test/test_qa/test_samples.json",
        # default="/home/wangshu/multimodal/GBC-RAG/test/sf/case-qa/sel_data_qa.json",
    )
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )

    args = parser.parse_args()
    cfg = load_system_config(args.config_path)
    inference_base(cfg, args.dataset_path)
