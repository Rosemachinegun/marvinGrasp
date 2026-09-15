"""Policy-independent dual-arm coordinate helpers."""

from __future__ import annotations


def normalize_hand(hand: str) -> str:
    value = str(hand).strip().lower()
    if value not in {"left", "right"}:
        raise ValueError(f"unsupported hand: {hand!r}")
    return value


def mirror_right_xyz_for_hand(
    right_xyz: tuple[float, float, float],
    hand: str,
) -> tuple[float, float, float]:
    x, y, z = (float(value) for value in right_xyz)
    return x, abs(y) if normalize_hand(hand) == "left" else -abs(y), z
