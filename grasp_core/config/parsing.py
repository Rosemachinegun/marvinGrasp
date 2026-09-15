"""Small, reusable parsers used by configuration loading."""

from __future__ import annotations

import argparse

import numpy as np


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected TRUE/FALSE, got {value!r}")


def parse_optional_bool(value: str | bool | None) -> bool | None:
    return None if value is None else parse_bool(value)


def parse_float_tuple(
    value: object,
    *,
    expected_len: int,
    fallback: tuple[float, ...],
    name: str,
) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        print(f"[grasp_config] invalid {name}={value!r}; using {fallback}", flush=True)
        return fallback
    if len(values) != expected_len or not all(np.isfinite(values)):
        print(f"[grasp_config] invalid {name}={value!r}; using {fallback}", flush=True)
        return fallback
    return values


def parse_config_bool(value: object, *, fallback: bool, name: str) -> bool:
    try:
        return parse_bool(value)  # type: ignore[arg-type]
    except argparse.ArgumentTypeError:
        print(f"[grasp_config] invalid {name}={value!r}; using {fallback}", flush=True)
        return fallback


def parse_config_float(value: object, *, fallback: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        print(f"[grasp_config] invalid {name}={value!r}; using {fallback}", flush=True)
        return fallback
    if not np.isfinite(result):
        print(f"[grasp_config] invalid {name}={value!r}; using {fallback}", flush=True)
        return fallback
    return result
