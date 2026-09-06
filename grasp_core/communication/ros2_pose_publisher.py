#!/usr/bin/env python3
"""通信底座：唯一直接创建 ROS2 Node 并发布目标位姿/轨迹消息的模块。"""

from __future__ import annotations

import time

import numpy as np

from grasp_core.core.pose_math import checked_position, normalize_quaternion


class Ros2PoseTargetPublisher:
    """Low-level ROS2 publisher for left/right IK target topics."""

    def __init__(
        self,
        *,
        left_topic: str,
        right_topic: str,
        left_trajectory_topic: str | None = None,
        right_trajectory_topic: str | None = None,
        frame_id: str,
        node_name: str = "flowpose_request_ik_tester",
    ) -> None:
        try:
            import rclpy
            from builtin_interfaces.msg import Duration
            from geometry_msgs.msg import PoseStamped, Transform, Twist
            from rclpy.node import Node
            from trajectory_msgs.msg import MultiDOFJointTrajectory
            from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 Python packages are not importable. Source your ROS2 workspace first."
            ) from exc

        self.rclpy = rclpy
        self.Duration = Duration
        self.MultiDOFJointTrajectory = MultiDOFJointTrajectory
        self.MultiDOFJointTrajectoryPoint = MultiDOFJointTrajectoryPoint
        self.PoseStamped = PoseStamped
        self.Transform = Transform
        self.Twist = Twist
        self.frame_id = frame_id
        self.left_topic = left_topic
        self.right_topic = right_topic
        self.left_trajectory_topic = left_trajectory_topic
        self.right_trajectory_topic = right_trajectory_topic

        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node(node_name)
        self.left_pub = self.node.create_publisher(PoseStamped, left_topic, 10)
        self.right_pub = self.node.create_publisher(PoseStamped, right_topic, 10)
        self.left_trajectory_pub = (
            self.node.create_publisher(
                MultiDOFJointTrajectory,
                left_trajectory_topic,
                10,
            )
            if left_trajectory_topic
            else None
        )
        self.right_trajectory_pub = (
            self.node.create_publisher(
                MultiDOFJointTrajectory,
                right_trajectory_topic,
                10,
            )
            if right_trajectory_topic
            else None
        )

    def wait_for_settled_tool_pose(
        self, hand: str, *, timeout_sec: float = 5.0, cancelled=lambda: False,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        from grasp_core.communication.measured_home_pose import wait_for_measured_home_pose

        return wait_for_measured_home_pose(
            self, hand, timeout_sec=timeout_sec, cancelled=cancelled,
        )

    def close(self) -> None:
        self.node.destroy_node()

    def ok(self) -> bool:
        return bool(self.rclpy.ok())

    def topic_for_hand(self, hand: str) -> str:
        return self.left_topic if hand == "left" else self.right_topic

    def trajectory_topic_for_hand(self, hand: str) -> str | None:
        return (
            self.left_trajectory_topic
            if hand == "left"
            else self.right_trajectory_topic
        )

    def trajectory_subscriber_count(self, hand: str) -> int:
        topic = self.trajectory_topic_for_hand(hand)
        if topic is None:
            return 0
        return int(self.node.count_subscribers(topic))

    def publish_pose(
        self,
        hand: str,
        position_xyz: np.ndarray,
        orientation_xyzw: tuple[float, float, float, float],
    ) -> None:
        publisher = self.left_pub if hand == "left" else self.right_pub
        publisher.publish(self.make_pose_stamped(position_xyz, orientation_xyzw))

    def publish_pose_trajectory(
        self,
        hand: str,
        samples: list[tuple[np.ndarray, tuple[float, float, float, float]]],
        *,
        joint_name: str,
        sample_period_sec: float,
        sample_periods_sec: list[float] | None = None,
    ) -> int:
        publisher = (
            self.left_trajectory_pub if hand == "left" else self.right_trajectory_pub
        )
        topic = self.trajectory_topic_for_hand(hand)
        if publisher is None or topic is None:
            raise RuntimeError(f"trajectory topic is not configured for hand={hand!r}")
        if not samples:
            return 0

        period_sec = max(float(sample_period_sec), 1e-4)
        if sample_periods_sec is None:
            periods_sec = [period_sec] * len(samples)
        else:
            periods_sec = [
                max(float(value), 1e-4) for value in sample_periods_sec[: len(samples)]
            ]
            if len(periods_sec) < len(samples):
                periods_sec.extend([period_sec] * (len(samples) - len(periods_sec)))
        msg = self.MultiDOFJointTrajectory()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.joint_names = [str(joint_name)]

        positions = [
            checked_position(position_xyz).copy() for position_xyz, _ in samples
        ]
        velocities = trajectory_linear_velocities(positions, periods_sec)
        elapsed_sec = 0.0
        for index, (position_xyz, orientation_xyzw) in enumerate(samples):
            point = self.MultiDOFJointTrajectoryPoint()
            point_period_sec = periods_sec[index]
            elapsed_sec += point_period_sec
            point.transforms = [self.make_transform(position_xyz, orientation_xyzw)]
            point.velocities = [self.make_linear_twist(velocities[index])]
            point.time_from_start = self.duration_from_seconds(elapsed_sec)
            msg.points.append(point)

        publisher.publish(msg)
        return len(msg.points)

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

    def make_transform(
        self,
        position_xyz: np.ndarray,
        orientation_xyzw: tuple[float, float, float, float],
    ):
        position = checked_position(position_xyz)
        qx, qy, qz, qw = normalize_quaternion(orientation_xyzw)
        transform = self.Transform()
        transform.translation.x = float(position[0])
        transform.translation.y = float(position[1])
        transform.translation.z = float(position[2])
        transform.rotation.x = qx
        transform.rotation.y = qy
        transform.rotation.z = qz
        transform.rotation.w = qw
        return transform

    def make_linear_twist(self, linear_xyz: np.ndarray):
        values = checked_position(linear_xyz)
        twist = self.Twist()
        twist.linear.x = float(values[0])
        twist.linear.y = float(values[1])
        twist.linear.z = float(values[2])
        return twist

    def duration_from_seconds(self, seconds: float):
        value = max(float(seconds), 0.0)
        sec = int(value)
        nanosec = int(round((value - sec) * 1_000_000_000))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        duration = self.Duration()
        duration.sec = sec
        duration.nanosec = nanosec
        return duration


def trajectory_linear_velocities(
    positions: list[np.ndarray],
    periods_sec: list[float],
) -> list[np.ndarray]:
    """Estimate waypoint velocities with zero start/end velocity boundaries."""

    if not positions:
        return []
    if len(positions) == 1:
        return [np.zeros(3, dtype=np.float64)]

    velocities: list[np.ndarray] = []
    for index in range(len(positions)):
        if index == 0 or index == len(positions) - 1:
            velocities.append(np.zeros(3, dtype=np.float64))
            continue

        previous_position = checked_position(positions[index - 1])
        next_position = checked_position(positions[index + 1])
        previous_dt = max(float(periods_sec[index]), 1e-4)
        next_dt = max(float(periods_sec[index + 1]), 1e-4)
        velocities.append(
            (next_position - previous_position) / (previous_dt + next_dt)
        )

    return velocities
