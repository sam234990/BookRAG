# In Core/utils/file_utils.py

import json
import logging
from pathlib import Path
from typing import Dict, Any

log = logging.getLogger(__name__)


def save_indexing_stats(save_path: str, new_stats: Dict[str, Any]):
    """
    Intelligently merges new run statistics with existing stats and saves to a JSON file.

    - For keys like 'build_tree_time' and 'build_kg_time', it only adds them if they don't already exist.
    - For 'token_stage_history', it merges the new stages without overwriting existing ones.

    :param save_path: The base directory where the stats file is located.
    :param new_stats: A dictionary containing the new stats from the current run.
    """
    stats_file = Path(save_path) / "indexing_stats.json"

    # 1. Load existing data if the file already exists
    if stats_file.exists():
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                final_stats = json.load(f)
        except json.JSONDecodeError:
            log.warning(
                f"Could not decode {stats_file}. A new stats file will be created."
            )
            final_stats = {}
    else:
        final_stats = {}

    # 2. Intelligently merge new stats into the final dictionary

    # Handle timing fields: only add if not present
    for time_key in ["build_tree_time", "build_kg_time"]:
        if time_key in new_stats and time_key not in final_stats:
            final_stats[time_key] = new_stats[time_key]

    # Handle token stage history: merge dictionaries without overwriting
    if "token_stage_history" in new_stats:
        if "token_stage_history" not in final_stats:
            final_stats["token_stage_history"] = {}
        # Iterate through each stage in the new stats
        for stage_name, new_stage_data in new_stats["token_stage_history"].items():
            # Case 1: The stage is completely new, so add it directly.
            if stage_name not in final_stats["token_stage_history"]:
                final_stats["token_stage_history"][stage_name] = new_stage_data
            else:
                # Case 2: The stage already exists, so we need to update its values.
                # Get the existing data for this stage.
                old_stage_data = final_stats["token_stage_history"][stage_name]
                
                # Iterate through each token key ("prompt_tokens", "completion_tokens", etc.)
                for token_key, new_value in new_stage_data.items():
                    # Get the old value, defaulting to 0 if it doesn't exist for some reason.
                    old_value = old_stage_data.get(token_key, 0)
                    
                    # YOUR CORE LOGIC: Update only if the new value is not 0 and is different.
                    if new_value != 0 and new_value != old_value:
                        final_stats["token_stage_history"][stage_name][token_key] = new_value


    # 3. Write the final, merged data back to the file
    try:
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(final_stats, f, indent=4)
        log.info(f"Successfully updated indexing stats at {stats_file}")
    except Exception as e:
        log.error(f"Failed to save indexing stats to {stats_file}. Error: {e}")
