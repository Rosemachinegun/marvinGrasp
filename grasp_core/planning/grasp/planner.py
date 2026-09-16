"""统一的抓取规划入口。

该类只负责策略编排和目标计算；轨迹发布、ROS 通信和夹爪控制仍由
execution 层负责。
"""

from __future__ import annotations

import argparse
from copy import copy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import yaml

from grasp_core.core.math.pose import (
    apply_downward_end_effector_tilt,
    apply_grasp_rotation_mode,
    apply_grasp_tcp_offset,
    apply_pregrasp_offset,
    build_grasp_pose,
    checked_position,
    get_relative_grasp_template,
    ik_orientation_rotation,
    matrix_to_quaternion,
    make_fixed_front_gripper_target_pose,
    normalize_object_type,
    normalize_quaternion,
    pose_from_position_quaternion,
    PickTemplateWaypoint,
    PoseWaypoint,
    validate_pose_matrix,
)
from grasp_core.core.math.object_axes import classify_box_shape
from grasp_core.core.types.robot_target_pose import TargetObjectPose
from grasp_core.config.yaml_loader import load_tool_yaml
from grasp_core.planning.grasp.policies.long_object import build_cuboid_pick_waypoints
from grasp_core.planning.grasp.policies.long_object import (
    make_cuboid_gripper_pose,
)
from grasp_core.planning.trajectory.planner import build_smooth_waypoints


def resolve_grasp_pose(
    pose: np.ndarray,
    args: argparse.Namespace,
    *,
    hand: str | None,
    stage: str = "grasp",
) -> np.ndarray:
    """Apply common grasp-pose adjustments exactly once."""
    if stage not in {"grasp", "pregrasp"}:
        raise ValueError(f"unsupported grasp stage: {stage!r}")
    resolved = np.asarray(pose, dtype=np.float64).copy()
    resolved = apply_grasp_rotation_mode(resolved, args)
    resolved = apply_grasp_tcp_offset(resolved, args)
    if stage == "pregrasp":
        resolved = apply_pregrasp_offset(resolved, args)
    return apply_downward_end_effector_tilt(resolved, args, hand=hand)


class GraspShape(str, Enum):
    CUBE = "cube"
    CUBOID = "cuboid"


@dataclass(frozen=True)
class GraspPlan:
    """Planning result consumed by the execution layer."""

    approach_waypoints: list[PoseWaypoint]
    final_approach_waypoints: list[PoseWaypoint]
    grip_position: np.ndarray
    grip_orientation: tuple[float, float, float, float]
    final_position: np.ndarray
    final_orientation: tuple[float, float, float, float]
    gripper_pose: np.ndarray
    template: np.ndarray | None
    fallback_reason: str | None
    used_pick_template: bool
    grip_waypoint_index: int | None
    grip_required: bool
    grip_state: float | None


class UnifiedGraspPlanner:
    """统一管理 cube/cuboid、姿态计算和模板展开。"""

    _template_cache: dict[tuple[str, int, bool], dict] = {}

    def classify(self, target: TargetObjectPose) -> GraspShape | None:
        """Return ``cube`` or ``cuboid`` from geometry only."""
        value = classify_box_shape(target.size)
        return GraspShape(value) if value is not None else None

    def make_gripper_target_pose(
        self,
        target: TargetObjectPose,
        args: argparse.Namespace,
        *,
        hand: str | None = None,
    ):
        return _make_gripper_target_pose(target, args, hand=hand)

    def plan_pose(
        self,
        target: TargetObjectPose,
        args: argparse.Namespace,
        *,
        hand: str | None = None,
    ):
        """Resolve the final computed gripper pose for one target."""
        return self.make_gripper_target_pose(target, args, hand=hand)

    def load_templates(self, args: argparse.Namespace):
        """Load and cache templates by path, mtime and enable flag."""
        path = Path(args.tool_template_path).expanduser()
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            mtime_ns = -1
        key = (str(path), mtime_ns, bool(getattr(args, "use_tool_pick_template", True)))
        if key not in self._template_cache:
            self._template_cache[key] = load_tool_pick_templates(args)
        return self._template_cache[key]

    def plan_template_waypoints(
        self,
        target: TargetObjectPose,
        relative_waypoints: list[PickTemplateWaypoint],
        args: argparse.Namespace,
        *,
        hand: str | None = None,
    ) -> list[PickTemplateWaypoint]:
        """Expand template waypoints after the shape policy is resolved."""
        return build_pick_template_waypoints(
            target,
            relative_waypoints,
            args,
            hand=hand,
        )

    def build_waypoints(
        self,
        start_position: np.ndarray,
        start_orientation: tuple[float, float, float, float],
        end_position: np.ndarray,
        end_orientation: tuple[float, float, float, float],
        args: argparse.Namespace,
    ) -> list[PoseWaypoint]:
        """Build the common approach trajectory for a planned grasp."""
        return build_smooth_waypoints(
            start_position,
            start_orientation,
            end_position,
            end_orientation,
            args,
            lift_arc=False,
        )

    @staticmethod
    def build_final_approach_waypoints(
        start_position: np.ndarray,
        start_orientation: tuple[float, float, float, float],
        end_position: np.ndarray,
        end_orientation: tuple[float, float, float, float],
    ) -> list[PoseWaypoint]:
        """Return the final target; slowdown is applied by the trajectory planner."""
        del start_position, start_orientation
        return [(checked_position(end_position).copy(), normalize_quaternion(end_orientation))]

    def grip_waypoint_index(
        self,
        waypoints: list[PickTemplateWaypoint],
    ) -> int | None:
        if not waypoints:
            return None
        candidates = [
            index
            for index, (_position, _orientation, gripper_state) in enumerate(waypoints)
            if float(gripper_state) >= 0.5
        ]
        if candidates:
            return min(candidates, key=lambda index: float(waypoints[index][0][2]))
        fallback_index = min(
            range(len(waypoints)),
            key=lambda index: float(waypoints[index][0][2]),
        )
        position = checked_position(waypoints[fallback_index][0])
        print(
            "[tool_template] WARNING no gripper_state>=0.5 in pick template; "
            "defaulting grip trigger to lowest waypoint "
            f"index={fallback_index} "
            f"xyz=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f})",
            flush=True,
        )
        return fallback_index

    def build_template_plan(
        self,
        target: TargetObjectPose,
        relative_pick_waypoints: list[PickTemplateWaypoint],
        start_position: np.ndarray,
        start_orientation: tuple[float, float, float, float],
        args: argparse.Namespace,
        hand: str,
    ) -> GraspPlan:
        pick_waypoints = self.plan_template_waypoints(
            target, relative_pick_waypoints, args, hand=hand
        )
        pose_waypoints = [
            (position, orientation)
            for position, orientation, _gripper_state in pick_waypoints
        ]
        grip_index = self.grip_waypoint_index(pick_waypoints)
        grip_state = (
            float(pick_waypoints[grip_index][2]) if grip_index is not None else None
        )
        if grip_index is None:
            final_position, final_orientation = pose_waypoints[-1]
            approach_waypoints = self.build_waypoints(
                start_position, start_orientation, final_position, final_orientation, args
            )
            grip_position, grip_orientation = final_position, final_orientation
            final_approach_waypoints: list[PoseWaypoint] = []
        else:
            grip_position, grip_orientation = pose_waypoints[grip_index]
            if grip_index > 0:
                pregrasp_position, pregrasp_orientation = pose_waypoints[grip_index - 1]
                approach_waypoints = self.build_waypoints(
                    start_position, start_orientation,
                    pregrasp_position, pregrasp_orientation, args
                )
                final_approach_waypoints = self.build_final_approach_waypoints(
                    pregrasp_position, pregrasp_orientation,
                    grip_position, grip_orientation
                )
            else:
                approach_waypoints = self.build_waypoints(
                    start_position, start_orientation, grip_position, grip_orientation, args
                )
                final_approach_waypoints = []
            final_position, final_orientation = grip_position, grip_orientation
        return GraspPlan(
            approach_waypoints=approach_waypoints,
            final_approach_waypoints=final_approach_waypoints,
            grip_position=checked_position(grip_position).copy(),
            grip_orientation=normalize_quaternion(grip_orientation),
            final_position=checked_position(final_position).copy(),
            final_orientation=normalize_quaternion(final_orientation),
            gripper_pose=pose_from_position_quaternion(final_position, final_orientation),
            template=None,
            fallback_reason=None,
            used_pick_template=True,
            grip_waypoint_index=grip_index,
            grip_required=grip_index is not None,
            grip_state=grip_state,
        )

    def build_computed_plan(
        self,
        target: TargetObjectPose,
        start_position: np.ndarray,
        start_orientation: tuple[float, float, float, float],
        args: argparse.Namespace,
        hand: str,
    ) -> GraspPlan:
        pregrasp_pose, template, fallback_reason = self.plan_pose(
            target, args, hand=hand
        )
        grasp_args = copy(args)
        grasp_args.ik_target_stage = "grasp"
        gripper_pose, _grasp_template, _grasp_fallback_reason = self.plan_pose(
            target, grasp_args, hand=hand
        )
        pregrasp_position = pregrasp_pose[:3, 3].copy()
        pregrasp_orientation = matrix_to_quaternion(pregrasp_pose)
        position = gripper_pose[:3, 3].copy()
        orientation = matrix_to_quaternion(gripper_pose)
        final_approach_waypoints: list[PoseWaypoint] = []
        if args.ik_target_stage == "pregrasp":
            final_approach_waypoints = self.build_final_approach_waypoints(
                pregrasp_position, pregrasp_orientation, position, orientation
            )
        return GraspPlan(
            approach_waypoints=self.build_waypoints(
                start_position, start_orientation,
                pregrasp_position, pregrasp_orientation, args
            ),
            final_approach_waypoints=final_approach_waypoints,
            grip_position=position.copy(),
            grip_orientation=orientation,
            final_position=position.copy(),
            final_orientation=orientation,
            gripper_pose=gripper_pose,
            template=template,
            fallback_reason=fallback_reason,
            used_pick_template=False,
            grip_waypoint_index=None,
            grip_required=True,
            grip_state=None,
        )


def make_gripper_target_pose(
    target: TargetObjectPose,
    args: argparse.Namespace,
    hand: str | None = None,
):
    """Compatibility wrapper; new code should use UnifiedGraspPlanner."""
    return _make_gripper_target_pose(target, args, hand=hand)

def _make_gripper_target_pose(
    target: TargetObjectPose,
    args: argparse.Namespace,
    hand: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, str | None]:
    """Resolve a base-frame gripper pose using the unified shape pipeline."""
    long_result = None
    if classify_box_shape(target.size) == GraspShape.CUBOID.value:
        long_result = make_cuboid_gripper_pose(
            target, args, hand=hand or "", force_long_object=True
        )
    if long_result is not None:
        gripper_pose, _metadata = long_result
        return gripper_pose, None, "long_object_shape_policy"

    object_pose = np.asarray(target.base_pose, dtype=np.float64)
    fallback_reason = validate_pose_matrix(object_pose)
    template = None
    if fallback_reason is None:
        template = get_relative_grasp_template(target.label)
        if template is None:
            fallback_reason = (
                f"no grasp template configured for object_type={target.label!r}"
            )

    if fallback_reason is not None:
        grasp_args = copy(args)
        grasp_args.ik_target_stage = "grasp"
        grasp_args.ik_grasp_tcp_offset_m = (0.0, 0.0, 0.0)
        gripper_pose = make_fixed_front_gripper_target_pose(target, grasp_args)
        gripper_pose = resolve_grasp_pose(
            gripper_pose, args, hand=hand, stage=args.ik_target_stage
        )
        return gripper_pose, None, fallback_reason

    gripper_pose = resolve_grasp_pose(
        build_grasp_pose(object_pose, target.label),
        args,
        hand=hand,
        stage=args.ik_target_stage,
    )
    return gripper_pose, template, None


def load_tool_pick_templates(
    args: argparse.Namespace,
) -> dict[str, dict[str, list[PickTemplateWaypoint]]]:
    if not bool(getattr(args, "use_tool_pick_template", True)):
        print("[tool_template] disabled; using computed single grasp target", flush=True)
        return {}
    path = Path(args.tool_template_path).expanduser()
    try:
        raw = load_tool_yaml(path)
    except OSError as exc:
        print(f"[tool_template] unable to read {path}: {exc}; fallback enabled", flush=True)
        return {}
    except yaml.YAMLError as exc:
        print(f"[tool_template] invalid YAML {path}: {exc}; fallback enabled", flush=True)
        return {}
    templates_raw = raw.get("templates") if isinstance(raw, dict) else None
    if not isinstance(templates_raw, dict):
        print(f"[tool_template] no templates found in {path}; fallback enabled", flush=True)
        return {}
    templates: dict[str, dict[str, list[PickTemplateWaypoint]]] = {}
    for object_name, object_cfg in templates_raw.items():
        if not isinstance(object_cfg, dict) or not isinstance(object_cfg.get("pick"), list):
            continue
        object_templates: dict[str, list[PickTemplateWaypoint]] = {}
        for entry in object_cfg["pick"]:
            if not isinstance(entry, dict) or entry.get("action_name") != "pick":
                continue
            arm = str(entry.get("arm") or "").strip().lower()
            if arm not in {"left", "right"}:
                continue
            waypoints = parse_pick_waypoints(entry.get("pose_relative"), entry.get("gripper_state"))
            if waypoints:
                object_templates[arm] = waypoints
        if object_templates:
            templates[normalize_object_type(str(object_name))] = object_templates
    print(f"[tool_template] loaded {len(templates)} object pick template(s) from {path}", flush=True)
    return templates


def parse_pick_waypoints(pose_relative: object, gripper_state: object) -> list[PickTemplateWaypoint]:
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
        waypoints.append((values[:3].copy(), normalize_quaternion(tuple(values[3:7])), gripper_values[index]))
    return waypoints


def parse_gripper_states(gripper_state: object, waypoint_count: int) -> list[float]:
    defaults = [0.0] * max(int(waypoint_count), 0)
    if not isinstance(gripper_state, list):
        return defaults
    try:
        values = [float(value) for value in gripper_state]
    except (TypeError, ValueError):
        return defaults
    for index, value in enumerate(values[:len(defaults)]):
        defaults[index] = value
    return defaults


def pick_template_for_target(
    target: TargetObjectPose,
    hand: str,
    templates: dict[str, dict[str, list[PickTemplateWaypoint]]],
) -> list[PickTemplateWaypoint] | None:
    by_hand = templates.get(normalize_object_type(target.label))
    return None if by_hand is None else by_hand.get(hand)


def build_pick_template_waypoints(
    target: TargetObjectPose,
    relative_waypoints: list[PickTemplateWaypoint],
    args: argparse.Namespace,
    hand: str | None = None,
) -> list[PickTemplateWaypoint]:
    if classify_box_shape(target.size) == GraspShape.CUBOID.value:
        cuboid_waypoints = build_cuboid_pick_waypoints(
            target, relative_waypoints, args, hand=str(hand or ""), force_long_object=True
        )
        if cuboid_waypoints is not None:
            return cuboid_waypoints
    object_pose = np.asarray(target.base_pose, dtype=np.float64)
    fallback_reason = validate_pose_matrix(object_pose)
    if fallback_reason is not None:
        raise ValueError(f"invalid object pose for pick template: {fallback_reason}")
    waypoints: list[PickTemplateWaypoint] = []
    fixed_rotation = ik_orientation_rotation(args)
    for relative_xyz, _relative_quat, gripper_value in relative_waypoints:
        relative_point = np.ones(4, dtype=np.float64)
        relative_point[:3] = relative_xyz
        gripper_pose = np.eye(4, dtype=np.float64)
        gripper_pose[:3, 3] = (object_pose @ relative_point)[:3]
        gripper_pose[:3, :3] = fixed_rotation
        gripper_pose = resolve_grasp_pose(gripper_pose, args, hand=hand, stage="grasp")
        waypoints.append((gripper_pose[:3, 3].copy(), matrix_to_quaternion(gripper_pose), float(gripper_value)))
    return waypoints


__all__ = [
    "GraspPlan",
    "GraspShape",
    "UnifiedGraspPlanner",
    "make_gripper_target_pose",
    "resolve_grasp_pose",
    "load_tool_pick_templates",
    "pick_template_for_target",
    "build_pick_template_waypoints",
]
