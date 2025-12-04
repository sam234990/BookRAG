from Core.configs.dataset_config import DatasetConfig, load_dataset_config
from Eval.utils.m3doc_eval import eval_m3doc
from Eval.utils.mmlong_eval import eval_mmlong
from Eval.utils.qasper_eval import eval_qasper

import pandas as pd
import argparse


def create_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset_config",
        type=str,
        required=False,
        default="/home/wangshu/multimodal/GBC-RAG/Scripts/cfg/MMLongBench.yaml",
        help="Path to the dataset configuration file for batch processing.",
    )

    parser.add_argument(
        "--method",
        type=str,
        required=False,
        default="mmr",
        help="Method to use for evaluation (e.g., 'mmr', 'traverse').",
    )
    
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Number of parallel workers for processing.",
    )

    return parser.parse_args()


def eval(args):
    # Load the dataset
    data_cfg: DatasetConfig = load_dataset_config(args.dataset_config)
    data_df = pd.read_json(data_cfg.dataset_path)

    document_groups = data_df.groupby(["doc_uuid", "doc_path"])
    print(f"evaluation method: {args.method}")
    
    print(f"Total document groups: {len(document_groups)}")
    print(f"Total samples: {len(data_df)}")
    print(f"Dataset name: {data_cfg.dataset_name}")

    
    if data_cfg.dataset_name.lower() == "mmlongbench":
        eval_mmlong(data_df, data_cfg, args.method, max_workers=args.max_workers)
        print("MMLongBench dataset evaluation completed.")

    if data_cfg.dataset_name.lower() == "m3docrag":
        eval_m3doc(data_df, data_cfg, args.method, max_workers=args.max_workers)
        print("M3DocRAG dataset evaluation completed.")

    if data_cfg.dataset_name.lower() == "qasper":
        eval_qasper(data_df, data_cfg, args.method, max_workers=args.max_workers)
        print("QASPER dataset evaluation completed.")


if __name__ == "__main__":
    args = create_args()
    eval(args)
