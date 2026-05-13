"""Shared config loading utilities.

Supports an ``extends:`` key at the top level of a YAML file to inherit
from another config (resolved relative to the leaf config's directory).
Leaf-file keys deep-merge over the parent; mappings are merged recursively,
all other types (scalars, lists) are replaced wholesale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Union

import yaml


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], Mapping)
            and isinstance(value, Mapping)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config file, resolving an optional ``extends:`` chain."""
    path = Path(config_path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        return cfg

    parent = cfg.pop("extends", None)
    if parent is None:
        return cfg

    parent_path = (path.parent / parent).resolve()
    base = load_config(parent_path)
    if not isinstance(base, dict):
        raise ValueError(f"Parent config {parent_path} must be a mapping")
    return _deep_merge(base, cfg)
