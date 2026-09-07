from argparse import Namespace

import numpy as np

from grasp_core.tasks import robot_actions
from grasp_core.core.robot_target_pose import TargetObjectPose


class FakePutResult:
    status = "put complete"
    ok = True


def test_ribbon_grasp_min_limit_is_accepted_as_success(monkeypatch) -> None:
    monkeypatch.setattr(
        robot_actions,
        "publish_latest_request_ik_target",
        lambda *args: "GRIP_FAILED_MIN_LIMIT hand=right",
    )
    service = robot_actions.RobotActionService(
        args=Namespace(ik_hand="right", ik_target_index=0),
        ik_publisher=object(),
        pick_templates={},
    )
    identity = np.eye(4)

    result = service.publish_grasp(
        [
            TargetObjectPose(
                "yellow_ribbon_1",
                "yellow_ribbon_1",
                identity,
                identity,
            )
        ]
    )

    assert result.ok
    assert result.grasp_confirmed
    assert not result.failed_min_limit
    assert result.grasp_hand == "right"


def test_ribbon_put_skips_grasp_drop_detection(monkeypatch) -> None:
    baseline_calls = []
    monkeypatch.setattr(
        robot_actions,
        "read_grasp_baseline",
        lambda *args: baseline_calls.append(args),
    )
    monkeypatch.setattr(
        robot_actions,
        "execute_fixed_put_after_grasp",
        lambda *args, **kwargs: FakePutResult(),
    )
    service = robot_actions.RobotActionService(
        args=Namespace(ik_hand="right", grip_drop_detection=True, put_keep_pose=True),
        ik_publisher=object(),
        pick_templates={},
    )

    result = service.publish_put(
        grasp_confirmed=True,
        hand="right",
        object_label="Yellow_Ribbon_1",
    )

    assert baseline_calls == []
    assert result.ok
    assert result.object_label == "Yellow_Ribbon_1"


def test_non_ribbon_put_still_reads_drop_detection_baseline(monkeypatch) -> None:
    baseline_calls = []

    def read_baseline(*args):
        baseline_calls.append(args)
        return None

    monkeypatch.setattr(robot_actions, "read_grasp_baseline", read_baseline)
    monkeypatch.setattr(
        robot_actions,
        "execute_fixed_put_after_grasp",
        lambda *args, **kwargs: FakePutResult(),
    )
    service = robot_actions.RobotActionService(
        args=Namespace(ik_hand="right", grip_drop_detection=True, put_keep_pose=True),
        ik_publisher=object(),
        pick_templates={},
    )

    service.publish_put(
        grasp_confirmed=True,
        hand="right",
        object_label="toy",
    )

    assert len(baseline_calls) == 1
