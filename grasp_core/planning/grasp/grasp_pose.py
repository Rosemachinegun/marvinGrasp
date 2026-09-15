#!/usr/bin/env python3
"""规划层：根据物体位姿、模板和 TCP 偏置计算夹爪目标位姿。"""

from __future__ import annotations

import argparse

import numpy as np

from grasp_core.core.math.pose import (
    apply_downward_end_effector_tilt,
    apply_grasp_rotation_mode,
    apply_grasp_tcp_offset,
    apply_pregrasp_offset,
    build_grasp_pose,
    get_relative_grasp_template,
    make_fixed_front_gripper_target_pose,
    validate_pose_matrix,
)
from grasp_core.core.types.robot_target_pose import TargetObjectPose
from grasp_core.planning.grasp.policies.long_object import (
    make_screwdriver_handle_gripper_pose,
)


def make_gripper_target_pose(
    target: TargetObjectPose,
    args: argparse.Namespace,
    hand: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, str | None]:
    """Plan a base-frame gripper target pose for one perceived object."""

    screwdriver_result = make_screwdriver_handle_gripper_pose(
        target,
        args,
        hand=hand or "",
    )
    if screwdriver_result is not None:
        gripper_pose, _metadata = screwdriver_result
        return gripper_pose, None, "screwdriver_handle_z_yaw_policy"

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
        gripper_pose = make_fixed_front_gripper_target_pose(target, args)
        gripper_pose = apply_downward_end_effector_tilt(gripper_pose, args, hand=hand)
        return gripper_pose, None, fallback_reason

    gripper_pose = build_grasp_pose(object_pose, target.label)
    gripper_pose = apply_grasp_rotation_mode(gripper_pose, args)
    gripper_pose = apply_grasp_tcp_offset(gripper_pose, args)
    if args.ik_target_stage == "pregrasp":
        gripper_pose = apply_pregrasp_offset(gripper_pose, args)
    gripper_pose = apply_downward_end_effector_tilt(gripper_pose, args, hand=hand)
    return gripper_pose, template, None
