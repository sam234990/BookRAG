import os
import json
import pandas as pd
from Core.configs.dataset_config import DatasetConfig


def load_cost(res_dir: str):
    cost_path = os.path.join(res_dir, "token_cost.json")
    if os.path.exists(cost_path):
        with open(cost_path, "r") as f:
            costs = json.load(f)
        return costs
    return {}


def get_all_cost(data_df: pd.DataFrame, data_cfg: DatasetConfig, method: str):
    document_groups = data_df.groupby(["doc_uuid", "doc_path"])
    all_cost = []

    for (doc_uuid, doc_path), group in document_groups:
        dir_name = f"eval_{data_cfg.dataset_name}_{method}"
        doc_res_dir = os.path.join(data_cfg.working_dir, doc_uuid, dir_name)
        costs = load_cost(doc_res_dir)
        all_cost.append(costs)

    # Calculate total costs
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_time = 0.0

    for cost in all_cost:
        if cost:
            rag_cost = cost['rag_cost']
            if isinstance(rag_cost, dict):
                total_prompt_tokens += rag_cost.get("prompt_tokens", 0)
                total_completion_tokens += rag_cost.get("completion_tokens", 0)
                total_tokens += rag_cost.get("total_tokens", 0)
            elif isinstance(rag_cost, int):
                total_tokens += rag_cost  # In case rag_cost is just a float value
            elif isinstance(rag_cost, float):
                total_tokens += int(rag_cost)  # In case rag_cost is just a float value
            total_time += cost.get("time", 0.0)

    score_dict = {}
    # Add to score_dict
    score_dict["total_prompt_tokens"] = total_prompt_tokens
    score_dict["total_completion_tokens"] = total_completion_tokens
    score_dict["total_tokens"] = total_tokens
    score_dict["total_time"] = round(total_time, 6)

    print(f"Total tokens: {total_tokens}")
    print(f"Total time (s): {total_time:.2f}")

    return score_dict
