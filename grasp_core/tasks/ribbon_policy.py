"""Centralized grasp policy for objects whose labels contain ``ribbon``."""

from __future__ import annotations


def is_ribbon_object(object_label: str | None) -> bool:
    """Return whether an object label contains the ribbon keyword."""

    return "ribbon" in str(object_label or "").strip().casefold()


def assume_grasp_success(object_label: str | None) -> bool:
    """Ribbon grasps are accepted without contact or minimum-limit checks."""

    return is_ribbon_object(object_label)


def skip_grasp_drop_detection(object_label: str | None) -> bool:
    """Ribbon transport does not use gripper-position drop detection."""

    return is_ribbon_object(object_label)
