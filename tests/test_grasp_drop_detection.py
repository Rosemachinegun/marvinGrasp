from argparse import Namespace

from grasp_core.tasks.grasp_drop_detection import (
    GraspDropMonitor,
    feedback_position,
)


class FakePublisher:
    def __init__(self) -> None:
        self.stopped = False

    def request_stop(self) -> None:
        self.stopped = True


def test_feedback_position_parses_receiver_reply() -> None:
    assert feedback_position("Sent left gripper feedback: OK feedback pos=417 current=2") == 417
    assert feedback_position("feedback failed") is None


def test_monitor_trips_only_after_closing_more_than_target(monkeypatch) -> None:
    samples = iter(("feedback pos=470", "feedback pos=469"))
    monkeypatch.setattr(
        "grasp_core.tasks.grasp_drop_detection.send_gripper_signal",
        lambda *args, **kwargs: next(samples),
    )
    publisher = FakePublisher()
    monitor = GraspDropMonitor(
        Namespace(grip_drop_close_delta=30, grip_drop_poll_interval=0.02),
        "left",
        500,
        publisher,
    )
    monitor.start()
    assert monitor._thread is not None
    monitor._thread.join(timeout=0.2)
    monitor.close()

    assert monitor.dropped
    assert monitor.detected_position == 469
    assert publisher.stopped
