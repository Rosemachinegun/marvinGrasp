import numpy as np

from grasp_core.core.types.robot_target_pose import TargetObjectPose
from grasp_core.planning.grasp.target_order import (
    estimate_target_aabb,
    reorder_targets_for_grasp,
)


def make_target(label, xyz, size=(0.05, 0.1, 0.1)):
    pose = np.eye(4)
    pose[:3, 3] = xyz
    return TargetObjectPose(label, label, pose.copy(), pose, np.asarray(size))


def test_aabb_is_estimated_from_pose_and_size():
    box = estimate_target_aabb(make_target("box", (0.0, 0.0, 0.5), (0.2, 0.1, 0.1)))
    np.testing.assert_allclose(box.minimum, [-0.1, -0.05, 0.45])
    np.testing.assert_allclose(box.maximum, [0.1, 0.05, 0.55])


def test_singletons_are_ordered_by_nearest_base_link_distance():
    targets = [
        make_target("near", (0.10, 0.00, 0.40)),
        make_target("far", (0.30, 0.00, 0.50)),
        make_target("farthest", (0.50, 0.00, 0.60)),
    ]
    assert [target.label for target in reorder_targets_for_grasp(targets)] == [
        "near", "far", "farthest"
    ]


def test_singleton_precedes_multi_object_cluster():
    targets = [
        make_target("cluster_a", (0.50, 0.00, 0.50)),
        make_target("singleton", (0.10, 0.00, 0.50)),
        make_target("cluster_b", (0.56, 0.00, 0.50)),
    ]
    assert [target.label for target in reorder_targets_for_grasp(targets)] == [
        "singleton", "cluster_a", "cluster_b"
    ]


def test_oversized_targets_are_removed_before_ordering():
    targets = [
        make_target("valid", (0.10, 0.00, 0.50), size=(0.05, 0.1, 0.1)),
        make_target("oversized", (0.20, 0.00, 0.50), size=(0.051, 0.1, 0.1)),
    ]
    assert [target.label for target in reorder_targets_for_grasp(targets)] == [
        "valid"
    ]
