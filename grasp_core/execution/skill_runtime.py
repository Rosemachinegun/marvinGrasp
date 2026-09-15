"""Shared execution context for robot skills."""

from __future__ import annotations

import argparse

from grasp_core.communication.gripper_signal import send_gripper_signal
from grasp_core.execution.motion_executor import (
    RequestIkTargetPublisher,
    publish_home,
    publish_path,
    read_current_tool_pose,
    request_ik_publisher_unavailable_status,
)
from grasp_core.core.math.pose import PoseWaypoint
from grasp_core.execution.stop_control import publisher_stop_requested


class SkillRuntime:
    """Common dependencies and safety helpers used by motion skills."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        publisher: RequestIkTargetPublisher | None,
    ) -> None:
        self.args = args
        self.publisher = publisher

    def publisher_status(self) -> str | None:
        if self.publisher is not None:
            return None
        return request_ik_publisher_unavailable_status(self.args)

    def stopped(self) -> bool:
        return self.publisher is not None and publisher_stop_requested(self.publisher)

    def current_pose(self, hand: str) -> PoseWaypoint:
        if self.publisher is None:
            raise RuntimeError("request_ik publisher unavailable")
        return read_current_tool_pose(self.publisher, hand)

    def publish_path(self, hand: str, waypoints: list[PoseWaypoint], **kwargs) -> int:
        if self.publisher is None:
            return 0
        return publish_path(self.publisher, hand, waypoints, self.args, **kwargs)

    def publish_home(self, hand: str, home_xyz: tuple[float, float, float], **kwargs) -> str:
        return publish_home(self.publisher, hand, home_xyz, self.args, **kwargs)

    def send_gripper(self, command: str, hand: str | None = None) -> str:
        return send_gripper_signal(command, self.args, hand=hand)
