#!/usr/bin/env python3
"""Low-level ROS2 PoseStamped target transport for request IK."""

from __future__ import annotations

import time

import numpy as np

from grasp_core.core.math.pose import checked_position, normalize_quaternion


class Ros2PoseTargetPublisher:
    """Low-level ROS2 publisher for left/right IK target topics."""

    def __init__(
        self,
        *,
        left_topic: str,
        right_topic: str,
        frame_id: str,
        node_name: str = "flowpose_request_ik_tester",
    ) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import PoseStamped
            from rclpy.node import Node
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 Python packages are not importable. Source your ROS2 workspace first."
            ) from exc

        self.rclpy = rclpy
        self.PoseStamped = PoseStamped
        self.frame_id = frame_id
        self.left_topic = left_topic
        self.right_topic = right_topic

        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node(node_name)
        self.left_pub = self.node.create_publisher(PoseStamped, left_topic, 10)
        self.right_pub = self.node.create_publisher(PoseStamped, right_topic, 10)

    def wait_for_settled_tool_pose(
        self, hand: str, *, timeout_sec: float = 5.0, cancelled=lambda: False,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        from grasp_core.communication.request_ik_feedback import wait_for_measured_home_pose

        return wait_for_measured_home_pose(
            self, hand, timeout_sec=timeout_sec, cancelled=cancelled,
        )

    def wait_for_current_tool_pose(
        self, hand: str, *, timeout_sec: float = 5.0, cancelled=lambda: False,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        from grasp_core.communication.request_ik_feedback import wait_for_measured_home_pose

        return wait_for_measured_home_pose(
            self,
            hand,
            timeout_sec=timeout_sec,
            cancelled=cancelled,
            require_settled=False,
        )

    def close(self) -> None:
        measured_reader = getattr(self, "_measured_pose_reader", None)
        if measured_reader is not None:
            measured_reader.close()
            self._measured_pose_reader = None
        self.node.destroy_node()

    def ok(self) -> bool:
        return bool(self.rclpy.ok())

    def topic_for_hand(self, hand: str) -> str:
        return self.left_topic if hand == "left" else self.right_topic

    def publish_pose(
        self,
        hand: str,
        position_xyz: np.ndarray,
        orientation_xyzw: tuple[float, float, float, float],
    ) -> None:
        publisher = self.left_pub if hand == "left" else self.right_pub
        publisher.publish(self.make_pose_stamped(position_xyz, orientation_xyzw))

    def hold_pose(
        self,
        hand: str,
        position_xyz: np.ndarray,
        orientation_xyzw: tuple[float, float, float, float],
        *,
        duration_sec: float,
        publish_rate_hz: float,
    ) -> int:
        duration_sec = max(float(duration_sec), 0.0)
        if duration_sec <= 0.0:
            return 0

        period_sec = 1.0 / max(float(publish_rate_hz), 0.1)
        deadline = time.monotonic() + duration_sec
        count = 0
        while self.ok() and time.monotonic() < deadline:
            self.publish_pose(hand, position_xyz, orientation_xyzw)
            count += 1
            time.sleep(period_sec)
        return count

    def make_pose_stamped(
        self,
        position_xyz: np.ndarray,
        orientation_xyzw: tuple[float, float, float, float],
    ):
        position = checked_position(position_xyz)
        qx, qy, qz, qw = normalize_quaternion(orientation_xyzw)
        msg = self.PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg
