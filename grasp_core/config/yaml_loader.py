"""Public YAML loading entry points."""

from pathlib import Path

import yaml


def load_tool_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return raw if isinstance(raw, dict) else {}
