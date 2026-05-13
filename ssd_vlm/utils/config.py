"""Shared config loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config file and return its contents as a dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
