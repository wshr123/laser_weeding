"""Configuration helpers for the laser weeding project."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG_PATH = os.path.join(PACKAGE_ROOT, "cam_params.yaml")


def resolve_config_path(config_file: Optional[str]) -> str:
    """Resolve configuration path relative to repository root."""

    if not config_file:
        return DEFAULT_CONFIG_PATH

    expanded = os.path.expanduser(config_file)
    if os.path.isabs(expanded):
        return expanded

    candidate = os.path.join(PACKAGE_ROOT, expanded)
    if os.path.exists(candidate):
        return candidate

    return os.path.abspath(expanded)


def load_yaml_file(file_path: str) -> Dict[str, Any]:
    """Load a YAML file and return its dictionary representation."""

    with open(file_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
