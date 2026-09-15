#!/usr/bin/env python3
"""规划层：读取 resources/tool.yaml 中的 pick waypoint 模板并展开到 base 坐标系。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

import numpy as np
import yaml

from grasp_core.core.types.robot_target_pose import TargetObjectPose, matrix_to_quaternion
from grasp_core.planning.grasp.policies.long_object import (
    build_screwdriver_handle_pick_waypoints,
)
from grasp_core.config.yaml_loader import load_tool_yaml
from grasp_core.core.math.pose import (
    PickTemplateWaypoint,
    PoseWaypoint,
    ik_wrist_orientation_quat,
    normalize_object_type,
    normalize_quaternion,
    quaternion_to_rotation_matrix,
    validate_pose_matrix,
)

def load_tool_pick_templates(
    args: argparse.Namespace,
) -> dict[str, dict[str, list[PickTemplateWaypoint]]]:
    if not bool(getattr(args, "use_tool_pick_template", True)):
        print(
            "[tool_template] disabled; using computed single grasp target", flush=True
        )
        return {}

    path = Path(args.tool_template_path).expanduser()
    try:
        raw = load_tool_yaml(path)
    except OSError as exc:
        print(
            f"[tool_template] unable to read {path}: {exc}; fallback enabled",
            flush=True,
        )
        return {}
    except yaml.YAMLError as exc:
        print(
            f"[tool_template] invalid YAML {path}: {exc}; fallback enabled", flush=True
        )
        return {}

    templates_raw = raw.get("templates") if isinstance(raw, dict) else None
    if not isinstance(templates_raw, dict):
        print(
            f"[tool_template] no templates found in {path}; fallback enabled",
            flush=True,
        )
        return {}

    templates: dict[str, dict[str, list[PickTemplateWaypoint]]] = {}
    for object_name, object_cfg in templates_raw.items():
        if not isinstance(object_cfg, dict):
            continue
        pick_entries = object_cfg.get("pick")
        if not isinstance(pick_entries, list):
            continue
        object_templates: dict[str, list[PickTemplateWaypoint]] = {}
        for entry in pick_entries:
            if not isinstance(entry, dict) or entry.get("action_name") != "pick":
                continue
            arm = str(entry.get("arm") or "").strip().lower()
            if arm not in {"left", "right"}:
                continue
            waypoints = parse_pick_waypoints(
                entry.get("pose_relative"),
                entry.get("gripper_state"),
            )
            if waypoints:
                object_templates[arm] = waypoints
        if object_templates:
            templates[normalize_object_type(str(object_name))] = object_templates

    print(
        f"[tool_template] loaded {len(templates)} object pick template(s) from {path}",
        flush=True,
    )
    return templates


def parse_pick_waypoints(
    pose_relative: object,
    gripper_state: object,
) -> list[PickTemplateWaypoint]:
    if not isinstance(pose_relative, list):
        return []
    gripper_values = parse_gripper_states(gripper_state, len(pose_relative))
    waypoints: list[PickTemplateWaypoint] = []
    for index, raw_waypoint in enumerate(pose_relative):
        try:
            values = np.asarray(raw_waypoint, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            continue
        waypoints.append(
            (
                values[:3].copy(),
                normalize_quaternion(tuple(values[3:7])),
                gripper_values[index],
            )
        )
    return waypoints


def parse_gripper_states(gripper_state: object, waypoint_count: int) -> list[float]:
    defaults = [0.0] * max(int(waypoint_count), 0)
    if not isinstance(gripper_state, list):
        return defaults
    try:
        values = [float(value) for value in gripper_state]
    except (TypeError, ValueError):
        return defaults
    for index, value in enumerate(values[: len(defaults)]):
        defaults[index] = value
    return defaults


def pick_template_for_target(
    target: TargetObjectPose,
    hand: str,
    templates: dict[str, dict[str, list[PickTemplateWaypoint]]],
) -> list[PickTemplateWaypoint] | None:
    by_hand = templates.get(normalize_object_type(target.label))
    if by_hand is None:
        return None
    return by_hand.get(hand)


def build_pick_template_waypoints(
    target: TargetObjectPose,
    relative_waypoints: list[PickTemplateWaypoint],
    args: argparse.Namespace,
    hand: str | None = None,
) -> list[PickTemplateWaypoint]:
    screwdriver_waypoints = build_screwdriver_handle_pick_waypoints(
        target,
        relative_waypoints,
        args,
        hand=str(hand or ""),
    )
    if screwdriver_waypoints is not None:
        return screwdriver_waypoints

    object_pose = np.asarray(target.base_pose, dtype=np.float64)
    fallback_reason = validate_pose_matrix(object_pose)
    if fallback_reason is not None:
        raise ValueError(f"invalid object pose for pick template: {fallback_reason}")

    waypoints: list[PickTemplateWaypoint] = []
    fixed_orientation = ik_wrist_orientation_quat(args, hand=hand)
    fixed_rotation = quaternion_to_rotation_matrix(fixed_orientation)
    for relative_xyz, _relative_quat, gripper_value in relative_waypoints:
        relative_point = np.ones(4, dtype=np.float64)
        relative_point[:3] = relative_xyz
        gripper_pose = np.eye(4, dtype=np.float64)
        gripper_pose[:3, 3] = (object_pose @ relative_point)[:3]
        gripper_pose[:3, :3] = fixed_rotation
        waypoints.append(
            (
                gripper_pose[:3, 3].copy(),
                matrix_to_quaternion(gripper_pose),
                float(gripper_value),
            )
        )
    return waypoints
