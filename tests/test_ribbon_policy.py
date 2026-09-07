from grasp_core.tasks.ribbon_policy import (
    assume_grasp_success,
    is_ribbon_object,
    skip_grasp_drop_detection,
)


def test_any_label_containing_ribbon_uses_policy() -> None:
    labels = ["ribbon", "ribbon_1", "yellow_ribbon", "Yellow_Ribbon_2"]

    for label in labels:
        assert is_ribbon_object(label)
        assert assume_grasp_success(label)
        assert skip_grasp_drop_detection(label)


def test_non_ribbon_label_does_not_use_policy() -> None:
    assert not is_ribbon_object("yellow_cable")
    assert not assume_grasp_success(None)
    assert not skip_grasp_drop_detection("toy")
