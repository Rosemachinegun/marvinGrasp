"""DBSCAN 聚类、目标选择和 OBB 可视化。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from grasp_core.core.types.robot_target_pose import TargetObjectPose

CLUSTER_EPS_M = 0.12
CLUSTER_USE_Z = False
SINGLETON_FIRST = True
DISTANCE_AXIS = "xy"
MAX_TARGET_VOLUME_M3 = 0.0005


@dataclass(frozen=True)
class TargetOBB:
    center: np.ndarray
    axes: np.ndarray
    extents: np.ndarray
    corners: np.ndarray


@dataclass(frozen=True)
class TargetAABB:
    minimum: np.ndarray
    maximum: np.ndarray


def estimate_target_aabb(target: TargetObjectPose) -> TargetAABB:
    """Return the axis-aligned bounds of one target OBB."""
    corners = estimate_target_obb(target).corners
    return TargetAABB(corners.min(axis=0), corners.max(axis=0))


@dataclass(frozen=True)
class ClusterSelection:
    cluster_id: int
    member_indices: tuple[int, ...]
    selected_index: int
    obb: TargetOBB


def estimate_target_obb(target: TargetObjectPose) -> TargetOBB:
    """Estimate the oriented bounding box of one target."""
    pose = np.asarray(target.base_pose, dtype=np.float64)
    center = pose[:3, 3].copy()
    axes = pose[:3, :3].copy()
    size = np.asarray(target.size, dtype=np.float64).reshape(-1) if target.size is not None else np.zeros(3)
    if size.size != 3 or not np.all(np.isfinite(size)) or np.any(size < 0):
        size = np.zeros(3, dtype=np.float64)
    half = size * 0.5
    signs = np.asarray(np.meshgrid([-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], indexing="ij")).reshape(3, -1).T
    corners = center + (signs * half) @ axes.T
    return TargetOBB(center=center, axes=axes, extents=size, corners=corners)


def _cluster_obb(targets: list[TargetObjectPose], indices: list[int]) -> TargetOBB:
    points = np.concatenate([estimate_target_obb(targets[index]).corners for index in indices])
    center = points.mean(axis=0)
    centered = points - center
    if len(points) >= 3 and np.linalg.norm(centered) > 1e-12:
        _values, axes = np.linalg.eigh(np.cov(centered, rowvar=False, bias=True))
        axes = axes[:, ::-1]
        if np.linalg.det(axes) < 0.0:
            axes[:, -1] *= -1.0
    else:
        axes = np.eye(3, dtype=np.float64)
    local = centered @ axes
    minimum, maximum = local.min(axis=0), local.max(axis=0)
    extents = maximum - minimum
    obb_center = center + ((minimum + maximum) * 0.5) @ axes.T
    half = extents * 0.5
    signs = np.asarray(np.meshgrid([-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], indexing="ij")).reshape(3, -1).T
    corners = obb_center + (signs * half) @ axes.T
    return TargetOBB(obb_center, axes, extents, corners)


def _dbscan_labels(points: np.ndarray, *, eps_m: float) -> np.ndarray:
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("target ordering requires scikit-learn for DBSCAN") from exc
    return DBSCAN(eps=float(eps_m), min_samples=1).fit_predict(points)


def _distance_from_base(center: np.ndarray, distance_axis: str) -> float:
    """Return distance from the robot base using the configured axes."""
    center = np.asarray(center, dtype=np.float64)
    if distance_axis == "xyz":
        return float(np.linalg.norm(center[:3]))
    return float(np.linalg.norm(center[:2]))


def _target_volume_m3(target: TargetObjectPose) -> float:
    """Return the estimated target volume, or zero when size is invalid."""
    if target.size is None:
        return 0.0
    size = np.asarray(target.size, dtype=np.float64).reshape(-1)
    if size.size != 3 or not np.all(np.isfinite(size)) or np.any(size < 0.0):
        return 0.0
    return float(np.prod(size))


def _save_cluster_artifacts(targets, selections, selected_index, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        print(f"[target_order] matplotlib unavailable; no plot saved: {exc}", flush=True)
        return
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab10")
    edges = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
    for item in selections:
        color = cmap(item.cluster_id % 10)
        points = np.asarray([targets[i].base_xyz for i in item.member_indices])
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=45)

        # Draw the OBB of every individual object.  The cluster OBB below is
        # dashed, so the two levels remain visually distinguishable.
        for target_index in item.member_indices:
            target = targets[target_index]
            target_obb = estimate_target_obb(target)
            target_color = "green" if target_index == selected_index else color
            target_line = 2.4 if target_index == selected_index else 0.9
            target_alpha = 1.0 if target_index == selected_index else 0.72
            for first, second in edges:
                axis.plot(
                    *zip(target_obb.corners[first], target_obb.corners[second]),
                    color=target_color,
                    linewidth=target_line,
                    alpha=target_alpha,
                )

            # Target labels already contain the stable zero-based instance
            # name shared with SAM3 (for example ``toy_0``). Do not append a
            # new index here, otherwise SAM3 and OBB names would diverge.
            name = target.label
            axis.text(
                *target.base_xyz,
                name,
                color=target_color,
                fontsize=8,
                ha="left",
                va="bottom",
            )

        # Dashed outline: OBB enclosing the complete DBSCAN cluster.
        for first, second in edges:
            axis.plot(
                *zip(item.obb.corners[first], item.obb.corners[second]),
                color=color,
                linewidth=1.2,
                linestyle="--",
                alpha=0.9,
            )
    chosen = targets[selected_index]
    axis.scatter(
        *chosen.base_xyz,
        color="green",
        edgecolors="black",
        linewidths=0.7,
        s=150,
        marker="*",
        label=f"selected grasp target: {chosen.label}",
        zorder=10,
    )
    axis.set_xlabel("X base_link (m)")
    axis.set_ylabel("Y base_link (m)")
    axis.set_zlabel("Z base_link (m)")
    axis.set_title("Target DBSCAN clusters and OBBs (base_link)")
    axis.legend(loc="best")
    figure.tight_layout()
    image_path = output_dir / f"{stem}_target_clusters_3d.png"
    figure.savefig(image_path, dpi=160)
    plt.close(figure)
    print(f"[target_order] saved {image_path}", flush=True)


def reorder_targets_for_grasp(
    targets: list[TargetObjectPose],
    *,
    output_dir: Path | None = None,
    stem: str = "latest",
    cluster_eps_m: float = CLUSTER_EPS_M,
    cluster_use_z: bool = CLUSTER_USE_Z,
    singleton_first: bool = SINGLETON_FIRST,
    distance_axis: str = DISTANCE_AXIS,
    max_target_volume_m3: float = MAX_TARGET_VOLUME_M3,
) -> list[TargetObjectPose]:
    """Order targets for grasping using DBSCAN and base_link proximity.

    Singleton clusters are handled first, ordered by nearest horizontal
    distance to the robot's ``base_link`` origin. Multi-object clusters follow;
    within each one, the target farthest from that cluster's center is first.
    """
    if not targets:
        return []
    retained_targets = []
    for target in targets:
        volume_m3 = _target_volume_m3(target)
        # Allow a tiny floating-point tolerance so an exact boundary value
        # such as 0.05 * 0.1 * 0.1 is retained as requested.
        if volume_m3 > float(max_target_volume_m3) + 1e-12:
            print(
                "\033[91m"
                "[target_order] filtered oversized target "
                f"{target.label}: volume={volume_m3:.6f}m3 "
                f"> limit={float(max_target_volume_m3):.6f}m3"
                "\033[0m",
                flush=True,
            )
            continue
        retained_targets.append(target)
    targets = retained_targets
    if not targets:
        return []
    if distance_axis not in {"xy", "xyz"}:
        raise ValueError(f"distance_axis must be 'xy' or 'xyz', got {distance_axis!r}")
    centers = np.asarray([target.base_xyz for target in targets], dtype=np.float64)
    cluster_points = centers[:, :3] if cluster_use_z else centers[:, :2]
    labels = _dbscan_labels(cluster_points, eps_m=cluster_eps_m)
    selections, groups = [], []
    for cluster_id in sorted(set(int(label) for label in labels)):
        group = [i for i, label in enumerate(labels) if int(label) == cluster_id]
        cluster_center = centers[group].mean(axis=0)
        selected = max(
            group,
            key=lambda i: _distance_from_base(centers[i] - cluster_center, distance_axis),
        )
        selections.append(ClusterSelection(cluster_id, tuple(group), selected, _cluster_obb(targets, group)))
        remaining = sorted(
            (i for i in group if i != selected),
            key=lambda i: (
                _distance_from_base(centers[i], distance_axis),
                -float(estimate_target_obb(targets[i]).corners[:, 2].max()),
                i,
            ),
        )
        groups.append([selected] + remaining)
    # Isolated objects have priority. Among isolated objects, nearest to the
    # robot in the horizontal base_link plane comes first. For equal distances,
    # use the OBB top height as a deterministic tie-breaker. Multi-object
    # clusters are ordered afterward using their selected (outermost) target.
    groups.sort(
        key=lambda group: (
            0 if singleton_first and len(group) == 1 else 1,
            _distance_from_base(centers[group[0]], distance_axis),
            -float(estimate_target_obb(targets[group[0]]).corners[:, 2].max()),
        )
    )
    ordered_indices = [i for group in groups for i in group]
    selected_index = ordered_indices[0]
    if output_dir is not None:
        _save_cluster_artifacts(targets, selections, selected_index, output_dir, stem)
    return [targets[i] for i in ordered_indices]


__all__ = [
    "TargetAABB",
    "TargetOBB",
    "estimate_target_aabb",
    "estimate_target_obb",
    "reorder_targets_for_grasp",
]
