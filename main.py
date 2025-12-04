import logging
import argparse
import pandas as pd
import yaml
from pathlib import Path
import time
import math
from rich.logging import RichHandler

from Core.configs.system_config import load_system_config, SystemConfig
from Core.configs.dataset_config import load_dataset_config, DatasetConfig
from Core.construct_index import (
    construct_GBC_index,
    construct_vdb,
    compute_mm_reranker,
    rebuild_graph_vdb,
)
from Core.inference import inference
from Core.provider.TokenTracker import TokenTracker

log = logging.getLogger(__name__)  # Get logger for main


def create_args():
    """
    Configures the command-line arguments for the project.

    This function sets up a main parser with global options (like config paths)
    and then adds subparsers for the two main modes of operation: 'index' and 'rag'.
    """
    # --- Main Parser ---
    # This parser handles global arguments that are common to all commands.
    parser = argparse.ArgumentParser(
        description="A command-line interface for building an index or running RAG inference.",
        formatter_class=argparse.RawTextHelpFormatter,  # For better help text formatting
    )

    # Global arguments required by both 'index' and 'rag' commands
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the main system configuration file (e.g., config/main_config.yaml).",
    )
    parser.add_argument(
        "-d",
        "--dataset_config",
        type=str,
        required=False,
        help="(Optional) Path to the dataset configuration file for batch processing.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for more verbose logging and error output.",
    )

    parser.add_argument(
        "--nsplit",
        type=int,
        default=2,
        help="The total number of splits for parallel processing.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=2,
        help="The current split number (1-indexed) to process.",
    )

    # --- Subparsers for Commands ---
    # This will hold the subparsers for 'index' and 'rag'
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # --- 'index' Command ---
    # The parser for the 'index' command
    parser_index = subparsers.add_parser(
        "index",
        help="Build an search index from the documents specified in the dataset config.",
    )
    # You can add arguments specific to indexing here if needed in the future
    # For example:
    # parser_index.add_argument("--force-rebuild", action="store_true", help="Force rebuilding the index if it already exists.")

    # --- 'rag' Command ---
    # The parser for the 'rag' (inference) command
    parser_rag = subparsers.add_parser(
        "rag", help="Run RAG inference using a pre-built index."
    )
    # You can add arguments specific to inference here
    # For example, to run a single query from the command line:
    parser_rag.add_argument(
        "-q",
        "--query",
        type=str,
        help="(Optional) A single query to run. If not provided, runs queries from the dataset config.",
    )

    # --- ADD THIS ARGUMENT ---
    # This allows the user to select a specific pipeline stage to execute.
    parser_index.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["tree", "graph", "vdb", "all", "mm_reranker", "rebuild_graph_vdb"],
        help="Specify which stage of the indexing pipeline to run: "
        "'tree' - Build and save the document tree only. "
        "'graph' - Build and save the knowledge graph (requires a tree). "
        "'vdb' - Build and save the vector database (requires a tree). "
        "'all' - Run all stages sequentially."
        "'mm_reranker' - Build and save the multi-modal reranker (requires a tree). "
        "'rebuild_graph_vdb' - Rebuild the graph and vector database (requires GBC Index).",
    )

    return parser.parse_args()


def build_index(config: SystemConfig, stage: str = "all", data_df: pd.DataFrame = None):
    log.info(
        f"  - build_index called. Indexing '{config.pdf_path}' into '{config.save_path}'"
    )

    # Stage 1: Build the Document Tree
    if stage in ["tree", "all"]:
        log.info("  - STAGE: Building Document Tree...")
        # This function should build the tree and save it to config.save_path
        construct_GBC_index(config, tree_only=True)

    # Stage 2: Build the Knowledge Graph
    if stage in ["graph", "all"]:
        log.info("  - STAGE: Building Knowledge Graph...")
        # This function should LOAD the pre-existing tree and then build/save the graph
        construct_GBC_index(config)

    # Stage 3: Build the Vector Database
    if stage in ["vdb", "all"]:
        log.info("  - STAGE: Building Vector Database...")
        # This function should LOAD the pre-existing tree and then build/save the VDB
        construct_vdb(config)

    if stage == "mm_reranker":
        log.info("  - STAGE: Building MM Reranker Embedding...")
        compute_mm_reranker(config, data_df)
    
    if stage == "rebuild_graph_vdb":
        log.info("  - STAGE: Rebuilding Graph VDB...")
        rebuild_graph_vdb(config)


def run_inference(config: SystemConfig, data_df: pd.DataFrame, dataset_name: str):
    log.info(f"  - run_inference called. Using index from '{config.save_path}'")
    inference(
        cfg=config,
        data_df=data_df,
        dataset_name=dataset_name,
    )


def setup_logging(save_path: str, config_to_log: SystemConfig):
    """
    Sets up the root logger to output to both a Rich console and a timestamped file.
    A new log file is created for each run inside the specified save_path.

    :param save_path: The base directory for the current run, where logs will be saved.
    :param config_to_log: The configuration object to be logged at the start.
    """
    log_dir = Path(save_path) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate a timestamped log file name
    log_file = log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"

    # It's better to get the root logger and configure it directly
    # than to use basicConfig, especially for re-configuration.
    root_logger = logging.getLogger()

    # Clear any existing handlers to prevent duplicate logging
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(logging.INFO)

    # --- Create Handlers ---
    # 1. RichHandler for beautiful console output
    console_handler = RichHandler(rich_tracebacks=True, show_path=False)

    # 2. FileHandler for saving logs to a file
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="[%X]"
    )
    file_handler.setFormatter(file_formatter)

    # --- Add Handlers to the Root Logger ---
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # --- Log the Initial Configuration ---
    # Get a logger instance for this setup function
    log = logging.getLogger("LoggerSetup")
    log.info(f"Logging initialized. Log file will be saved to: {log_file}")

    # Prettify the config output using yaml.dump
    config_dict = config_to_log.model_dump()
    config_yaml_string = yaml.dump(
        config_dict, allow_unicode=True, default_flow_style=False
    )
    log.info(f"--- Starting Run with Configuration ---\n{config_yaml_string}")


def process_resource(base_system_cfg: SystemConfig, args):
    # Ensure the nested models exist before assigning
    base_system_cfg.mineru.server_url = "http://localhost:30001"

    # For Graph construction
    # base_system_cfg.graph.reranker_config.api_base = "http://localhost:8010/v1"
    # base_system_cfg.llm.api_base = "http://10.26.1.21:8002/v1"
    base_system_cfg.llm.api_base = "http://localhost:8003/v1"

    if base_system_cfg.rag.strategy_config.strategy == "mmr":
        base_system_cfg.rag.strategy_config.vdb_config.embedding_config.device = (
            "cuda:4"
        )

    # For GBC inference
    if base_system_cfg.rag.strategy_config.strategy == "gbc":
        base_system_cfg.rag.strategy_config.mm_reranker_config.device = "cuda:3"

    if args.debug and base_system_cfg.rag.strategy_config.strategy == "gbc":
        base_system_cfg.rag.strategy_config.mm_reranker_config.device = "cuda:3"

    return base_system_cfg


def main():
    """
    The main function to run the script.
    """
    args = create_args()

    log.info("--- Arguments Loaded ---")
    log.info(f"Main Config Path: {args.config}")
    log.info(f"Selected Command: {args.command}")
    log.info("------------------------\n")

    base_system_cfg: SystemConfig = load_system_config(args.config)

    token_tracker = TokenTracker.get_instance()
    token_tracker.reset()

    if args.num % 2 == 0 or args.debug:
        # modify the device config for mineru, embedding, reranker
        # assign 30001 to a half of the num
        log.info(
            f"  - Split {args.num}: Overriding mineru.server_url to http://localhost:30001"
        )
        base_system_cfg = process_resource(base_system_cfg, args)

    if args.dataset_config:
        dataset_cfg: DatasetConfig = load_dataset_config(args.dataset_config)

        # 1. Load the entire dataset from the JSON file into a pandas DataFrame
        log.info(f"  - Loading dataset from: {dataset_cfg.dataset_path}")
        try:
            df = pd.read_json(dataset_cfg.dataset_path)
            print(f"Dataset shape: {df.shape}")
        except FileNotFoundError:
            log.error(f"ERROR: Dataset file not found at '{dataset_cfg.dataset_path}'")
            return
        except Exception as e:
            log.error(f" ERROR: Failed to parse JSON file. Reason: {e}")
            return

        # 2. Group by document identifiers to find unique documents
        document_groups = df.groupby(["doc_uuid", "doc_path"])
        print(f"  - Found {len(document_groups)} unique documents in the dataset.")

        # Convert groupby object to a list to allow slicing
        all_groups = list(document_groups)
        total_docs = len(all_groups)

        if args.num > args.nsplit or args.num <= 0:
            print(
                f"  - ERROR: --num ({args.num}) must be between 1 and --nsplit ({args.nsplit})."
            )
            return

        # Calculate the start and end index for the current split
        items_per_split = math.ceil(total_docs / args.nsplit)
        start_index = (args.num - 1) * items_per_split
        end_index = min(
            start_index + items_per_split, total_docs
        )  # Ensure we don't go past the end

        docs_to_process = all_groups[start_index:end_index]

        print(
            f" This worker (split {args.num}/{args.nsplit}) will process {len(docs_to_process)} documents (from index {start_index} to {end_index-1})."
        )

        # 3. Loop over each unique document group
        index_error_list = []
        rag_error_list = []

        for (doc_uuid, doc_path), group in docs_to_process:
            if args.debug and doc_uuid != "fe4f4a15-bc6c-5bf1-a21d-7fe10130b991":
                continue

            # a. Create a deep copy of the base config for this specific document run.
            current_config = base_system_cfg.model_copy(deep=True)

            # b. Dynamically set the paths based on the dataset's content
            # The doc_path from JSON is the source PDF
            pdf_full_path = Path(doc_path)
            output_full_path = Path(dataset_cfg.working_dir) / str(
                doc_uuid
            )  # working_dir + uuid = save_path

            current_config.pdf_path = str(pdf_full_path)
            current_config.save_path = str(output_full_path)

            setup_logging(
                save_path=current_config.save_path, config_to_log=current_config
            )

            log.info(f"--- Processing Document UUID: {doc_uuid} ---")
            log.info(f"  - PDF Path: {pdf_full_path}")
            log.info(f"  - Save Path: {output_full_path}")

            # d. Ensure output directory exists and save a config snapshot for reproducibility
            output_full_path.mkdir(parents=True, exist_ok=True)

            if args.command == "index":
                # For indexing, save the general run_config.yaml
                config_snapshot_path = output_full_path / "run_config.yaml"
                with open(config_snapshot_path, "w", encoding="utf-8") as f:
                    yaml.dump(current_config.model_dump(), f, allow_unicode=True)
                log.info(f"  - Saved index config snapshot to: {config_snapshot_path}")

                try:
                    data_df = group.reset_index(drop=True)
                    build_index(
                        config=current_config, stage=args.stage, data_df=data_df
                    )
                except Exception as e:
                    log.error(f"  - ERROR: Failed to build index. Reason: {e}")
                    index_error_list.append((doc_uuid, str(e)))

            elif args.command == "rag":
                # For RAG, create a strategy-specific config name
                # We assume the path to the strategy is like: cfg.rag.strategy_config.strategy
                rag_strategy = current_config.rag.strategy_config.strategy
                config_snapshot_filename = f"rag_config_{rag_strategy}.yaml"
                config_snapshot_path = output_full_path / config_snapshot_filename

                with open(config_snapshot_path, "w", encoding="utf-8") as f:
                    yaml.dump(current_config.model_dump(), f, allow_unicode=True)
                log.info(f"  - Saved RAG config snapshot to: {config_snapshot_path}")

                dataset_name = dataset_cfg.dataset_name
                data_df = group.reset_index(drop=True)
                try:
                    run_inference(
                        config=current_config,
                        data_df=data_df,
                        dataset_name=dataset_name,
                    )
                except Exception as e:
                    log.error(f"  - ERROR: Failed to run inference. Reason: {e}")
                    rag_error_list.append((doc_uuid, str(e)))

        # get the base directory of the script
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_name = dataset_cfg.dataset_name
        if index_error_list:
            error_log_path = (
                Path(base_dir) / f"{dataset_name}-index_error_split_{args.num}.txt"
            )
            with open(error_log_path, "w", encoding="utf-8") as f:
                for item in index_error_list:
                    f.write(f"{item}\n")
            log.info(
                f"  - Indexing completed with errors. See {error_log_path} for details."
            )

        if rag_error_list:
            error_log_path = (
                Path(base_dir) / f"{dataset_name}-rag_error_split_{args.num}.txt"
            )
            with open(error_log_path, "w", encoding="utf-8") as f:
                for item in rag_error_list:
                    f.write(f"{item}\n")
            log.info(
                f"  - RAG completed with errors. See {error_log_path} for details."
            )
        log.info(f"--- All documents in this split processed. ---\n")
    else:
        # SINGLE FILE MODE

        setup_logging(
            save_path=base_system_cfg.save_path, config_to_log=base_system_cfg
        )

        log.info(f"🚀 No dataset config provided. Starting SINGLE mode...")
        log.info(f"  - Using paths and settings from '{args.config}'")

        if args.command == "index":
            build_index(config=base_system_cfg)

        elif args.command == "rag":
            # Check if the --query argument was provided
            if args.query:
                log.info(f"  - Running inference on single query: '{args.query}'")
                # Note: run_inference expects a list of queries, so we wrap the single query in a list
                run_inference(config=base_system_cfg, queries=[args.query])
            else:
                # If no query is provided, print a helpful message and exit.
                log.error("  - ERROR: RAG command in single mode requires a query.")
                log.error("  - Please provide one with the -q/--query argument.")


if __name__ == "__main__":
    main()
