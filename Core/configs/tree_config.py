# Copyright (C) 2025-2026 Shu Wang
# SPDX-License-Identifier: Apache-2.0 OR AGPL-3.0-only

from dataclasses import dataclass

@dataclass
class TreeConfig:
    node_keywords: bool = True
    node_summary: bool = False
    use_vlm: bool = False
