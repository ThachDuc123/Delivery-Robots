"""Tiny YAML config loader with single-key ``defaults`` inheritance.

Each algo config (ppo.yaml/sac.yaml/td3.yaml) starts with::

    defaults: default.yaml

and only overrides the keys it needs. ``load_config`` deep-merges the override
on top of the base so the rest of the code always sees one fully-resolved dict.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str) -> Dict[str, Any]:
    """Load ``path`` and recursively merge any ``defaults:`` parent config."""
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    parent = cfg.pop("defaults", None)
    if parent:
        parent_path = parent if os.path.isabs(parent) else os.path.join(os.path.dirname(path), parent)
        base = load_config(parent_path)
        cfg = _deep_merge(base, cfg)
    return cfg


def default_config_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "default.yaml")
