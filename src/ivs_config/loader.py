"""YAML load and nested dict merge."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return a plain dict."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        msg = f"YAML root must be a mapping, got {type(data).__name__}"
        raise TypeError(msg)
    return data


def merge_config(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Deep-merge ``overrides`` into a copy of ``base``."""
    if not overrides:
        return deepcopy(dict(base))
    out = deepcopy(dict(base))
    for key, val in overrides.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, Mapping):
            out[key] = merge_config(out[key], val)
        else:
            out[key] = deepcopy(val) if isinstance(val, dict) else val
    return out


def merge_config_files(*paths: str | Path) -> dict[str, Any]:
    """Load several YAML files and deep-merge in order (later files override earlier)."""
    acc: dict[str, Any] = {}
    for path in paths:
        acc = merge_config(acc, load_config(path))
    return acc
