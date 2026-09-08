from argparse import Namespace

from grasp_core.communication.gripper_signal import gripper_receiver_args
from grasp_core.config.request_ik_config import GRIPPER_DEFAULTS


def dual_gripper_args() -> Namespace:
    return Namespace(
        dual_gripper=True,
        grip_signal_port=55660,
        left_gripper_server=GRIPPER_DEFAULTS.left_server,
        right_gripper_server=GRIPPER_DEFAULTS.right_server,
        gripper_server=GRIPPER_DEFAULTS.server,
        gripper_clamp_pos=GRIPPER_DEFAULTS.clamp_pos,
        gripper_open_pos=GRIPPER_DEFAULTS.open_pos,
        gripper_max_itinerary=GRIPPER_DEFAULTS.max_itinerary,
        gripper_speed_coe=GRIPPER_DEFAULTS.speed_coe,
        gripper_min_pos=GRIPPER_DEFAULTS.min_pos,
        gripper_max_pos=GRIPPER_DEFAULTS.max_pos,
        gripper_grip_speed=GRIPPER_DEFAULTS.grip_speed,
        gripper_grip_torque=GRIPPER_DEFAULTS.grip_torque,
        gripper_hold_torque=GRIPPER_DEFAULTS.hold_torque,
        gripper_current_threshold=GRIPPER_DEFAULTS.current_threshold,
        gripper_poll_interval=GRIPPER_DEFAULTS.poll_interval,
        gripper_contact_grace=GRIPPER_DEFAULTS.contact_grace,
        gripper_progress_epsilon=GRIPPER_DEFAULTS.progress_epsilon,
        gripper_stall_samples=GRIPPER_DEFAULTS.stall_samples,
        gripper_empty_grip_margin=GRIPPER_DEFAULTS.empty_grip_margin,
        gripper_target_pos_tolerance=GRIPPER_DEFAULTS.target_pos_tolerance,
        gripper_timeout=GRIPPER_DEFAULTS.timeout,
        gripper_grip_done_wait=GRIPPER_DEFAULTS.grip_done_wait,
        gripper_release_target=GRIPPER_DEFAULTS.release_target,
        gripper_release_speed=GRIPPER_DEFAULTS.release_speed,
        gripper_release_torque=GRIPPER_DEFAULTS.release_torque,
        gripper_release_wait=GRIPPER_DEFAULTS.release_wait,
        left_gripper_clamp_pos=GRIPPER_DEFAULTS.left_clamp_pos,
        left_gripper_open_pos=GRIPPER_DEFAULTS.left_open_pos,
        left_gripper_max_itinerary=GRIPPER_DEFAULTS.left_max_itinerary,
        left_gripper_speed_coe=GRIPPER_DEFAULTS.left_speed_coe,
        left_gripper_min_pos=GRIPPER_DEFAULTS.left_min_pos,
        left_gripper_max_pos=GRIPPER_DEFAULTS.left_max_pos,
        left_gripper_grip_speed=GRIPPER_DEFAULTS.left_grip_speed,
        left_gripper_grip_torque=GRIPPER_DEFAULTS.left_grip_torque,
        left_gripper_hold_torque=GRIPPER_DEFAULTS.left_hold_torque,
        left_gripper_current_threshold=GRIPPER_DEFAULTS.left_current_threshold,
        left_gripper_poll_interval=GRIPPER_DEFAULTS.left_poll_interval,
        left_gripper_contact_grace=GRIPPER_DEFAULTS.left_contact_grace,
        left_gripper_progress_epsilon=GRIPPER_DEFAULTS.left_progress_epsilon,
        left_gripper_stall_samples=GRIPPER_DEFAULTS.left_stall_samples,
        left_gripper_empty_grip_margin=GRIPPER_DEFAULTS.left_empty_grip_margin,
        left_gripper_target_pos_tolerance=(
            GRIPPER_DEFAULTS.left_target_pos_tolerance
        ),
        left_gripper_timeout=GRIPPER_DEFAULTS.left_timeout,
        left_gripper_grip_done_wait=GRIPPER_DEFAULTS.left_grip_done_wait,
        left_gripper_release_target=GRIPPER_DEFAULTS.left_release_target,
        left_gripper_release_speed=GRIPPER_DEFAULTS.left_release_speed,
        left_gripper_release_torque=GRIPPER_DEFAULTS.left_release_torque,
        left_gripper_release_wait=GRIPPER_DEFAULTS.left_release_wait,
    )


def test_dual_receiver_args_keep_left_and_right_grasp_limits_separate() -> None:
    endpoints = dict(gripper_receiver_args(dual_gripper_args()))

    left = endpoints["left"]
    right = endpoints["right"]

    assert left.gripper_hand == "left"
    assert right.gripper_hand == "right"
    assert left.gripper_server == GRIPPER_DEFAULTS.left_server
    assert right.gripper_server == GRIPPER_DEFAULTS.right_server
    assert left.gripper_empty_grip_margin == 150
    assert right.gripper_empty_grip_margin == 0
    assert left.gripper_min_pos == right.gripper_min_pos == 100


def test_default_grip_confirmation_is_low_latency() -> None:
    """The close-to-lift gate should add well below 0.2 s after motion stalls."""

    assert GRIPPER_DEFAULTS.left_grip_speed == 80
    assert GRIPPER_DEFAULTS.right_grip_speed == 80
    assert GRIPPER_DEFAULTS.left_poll_interval == 0.02
    assert GRIPPER_DEFAULTS.right_poll_interval == 0.02
    assert GRIPPER_DEFAULTS.left_contact_grace == 0.1
    assert GRIPPER_DEFAULTS.right_contact_grace == 0.1
    assert GRIPPER_DEFAULTS.left_stall_samples == 3
    assert GRIPPER_DEFAULTS.right_stall_samples == 3
    assert (
        GRIPPER_DEFAULTS.right_contact_grace
        + GRIPPER_DEFAULTS.right_poll_interval * GRIPPER_DEFAULTS.right_stall_samples
        < 0.2
    )
