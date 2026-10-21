from argparse import Namespace
from threading import Event, Thread
from unittest.mock import ANY, Mock

import numpy as np

import grasp_core.execution.skills.handover as handover_module
from grasp_core.execution.config import HOME_CONFIG
from grasp_core.execution.skills.handover import (
    HANDOVER_CONFIRM_TIMEOUT_SEC,
    LEFT_HANDOVER_XYZ,
    POST_RELEASE_HOLD_SEC,
    execute_handover_after_grasp,
    handover_xyz_for_hand,
)
from grasp_core.execution.result import FixedPlaceResult


def test_handover_tool_targets_are_mirrored() -> None:
    assert handover_xyz_for_hand("left") == LEFT_HANDOVER_XYZ
    x, y, z = LEFT_HANDOVER_XYZ
    assert handover_xyz_for_hand("right") == (x, -y, z)


def test_handover_waits_for_confirmation_then_releases_and_returns_home(
    monkeypatch,
) -> None:
    publisher = Mock()
    start_pose = (np.array([0.4, 0.2, 0.7]), (0.0, 0.0, 0.0, 1.0))
    publish_path = Mock(return_value=12)
    release = Mock(return_value="Sent left gripper release")
    sleep = Mock()
    publish_home = Mock(return_value="Published left home target")
    monkeypatch.setattr(
        handover_module,
        "read_current_tool_pose",
        Mock(return_value=start_pose),
    )
    monkeypatch.setattr(
        handover_module,
        "publisher_stop_requested",
        Mock(return_value=False),
    )
    monkeypatch.setattr(handover_module, "publish_path", publish_path)
    monkeypatch.setattr(handover_module, "send_gripper_signal", release)
    monkeypatch.setattr(handover_module.time, "sleep", sleep)
    monkeypatch.setattr(handover_module, "publish_home", publish_home)
    release_requested = Event()
    ready = Event()
    result_holder = []

    worker = Thread(
        target=lambda: result_holder.append(
            execute_handover_after_grasp(
                publisher,
                "left",
                Namespace(),
                grasp_confirmed=True,
                release_requested=release_requested,
                ready_event=ready,
            )
        )
    )
    worker.start()

    assert ready.wait(timeout=1.0)
    release.assert_not_called()
    release_requested.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    result = result_holder[0]
    assert result.ok is True
    assert publish_path.call_args.kwargs["final_hold_sec"] == 0.0
    release.assert_called_once_with("release", ANY, hand="left")
    sleep.assert_called_once_with(POST_RELEASE_HOLD_SEC)
    publish_home.assert_called_once_with(
        publisher,
        "left",
        HOME_CONFIG.position("left"),
        ANY,
        final_hold_sec=HOME_CONFIG.hold_sec,
    )


def test_handover_timeout_plays_prompt_then_uses_fixed_place(monkeypatch) -> None:
    assert HANDOVER_CONFIRM_TIMEOUT_SEC == 7.0
    publisher = Mock()
    start_pose = (np.array([0.4, 0.2, 0.7]), (0.0, 0.0, 0.0, 1.0))
    events = []
    timeout_prompt = Mock(side_effect=lambda: events.append("prompt"))
    fallback_place = Mock(
        side_effect=lambda *args, **kwargs: (
            events.append("place") or FixedPlaceResult(True, "place release; HOME")
        )
    )
    release = Mock()
    monkeypatch.setattr(handover_module, "read_current_tool_pose", Mock(return_value=start_pose))
    monkeypatch.setattr(handover_module, "publisher_stop_requested", Mock(return_value=False))
    monkeypatch.setattr(handover_module, "publish_path", Mock(return_value=12))
    monkeypatch.setattr(handover_module, "send_gripper_signal", release)
    monkeypatch.setattr(handover_module, "execute_fixed_place_after_grasp", fallback_place)
    monkeypatch.setattr(handover_module, "HANDOVER_CONFIRM_TIMEOUT_SEC", 0.0)
    monkeypatch.setattr(handover_module, "HANDOVER_CONFIRM_POLL_SEC", 0.0)
    ready = Event()

    result = execute_handover_after_grasp(
        publisher,
        "right",
        Namespace(),
        grasp_confirmed=True,
        release_requested=Event(),
        ready_event=ready,
        object_type="toy",
        timeout_prompt=timeout_prompt,
    )

    assert result.ok is True
    assert "timed out after 0s" in result.status
    assert events == ["prompt", "place"]
    assert ready.is_set() is False
    release.assert_not_called()
    fallback_place.assert_called_once_with(
        publisher,
        "right",
        ANY,
        grasp_confirmed=True,
        object_type="toy",
        target_count=1,
    )
