# Copyright (C) 2025-2026 Shu Wang
# SPDX-License-Identifier: Apache-2.0 OR AGPL-3.0-only

from pydantic import BaseModel, Field
import yaml


class DatasetConfig(BaseModel):
    dataset_path: str = Field(..., description="Path to the JSON dataset file.")
    working_dir: str = Field(..., description="The working directory for the project.")
    dataset_name: str


def load_dataset_config(path: str) -> DatasetConfig:
    # ... standard YAML loading logic ...
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    data_cfg = DatasetConfig(**data)
    return data_cfg
