"""
Visualization utilities for YomniFlow inference.
Exact port of cutoop's draw_3d_bbox / draw_pose_axes / draw_bboxes.
"""
import cv2
import numpy as np


# Exact copy of cutoop's CUBE_3x8 (shape [3, 8]) and _bbox_coef (shape [8, 3])
_CUBE_3x8 = np.array([
    [+1, +1, +1],
    [+1, +1, -1],
    [-1, +1, +1],
    [-1, +1, -1],
    [+1, -1, +1],
    [+1, -1, -1],
    [-1, -1, +1],
    [-1, -1, -1],
]).T  # [3, 8]

_bbox_coef = _CUBE_3x8.T  # [8, 3]  – same values as cutoop's _bbox_coef

_BBOX_LINES = [
    [4, 5], [5, 7], [6, 4], [7, 6],
    [0, 4], [1, 5], [2, 6], [3, 7],
    [0, 1], [1, 3], [2, 0], [3, 2],
]


# ── exact copies of cutoop internals ──────────────────────────────────────────

def _draw_seg(img, start, end, start_color, end_color=None, thickness_coef=1):
    if end_color is None:
        end_color = start_color
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    start_color = np.array(start_color, dtype=float)
    end_color = np.array(end_color, dtype=float)
    thickness = round(img.shape[1] / 320 * thickness_coef)
    steps = 10
    step = (end - start) / steps
    s_step = (end_color - start_color) / steps
    for i in range(steps):
        x, y = start + (i * step), start + ((i + 1) * step)
        sx = start_color + ((i + 0.5) * s_step)
        x = tuple(x.astype(int))
        y = tuple(y.astype(int))
        sx = tuple(sx.astype(int))
        img = cv2.line(
            np.ascontiguousarray(img.copy()),
            x, y,
            (int(sx[0]), int(sx[1]), int(sx[2])),
            thickness,
        )
    return img


def _create_3d_bbox(size):
    """Exact copy of cutoop's create_3d_bbox. Returns [3, 8]."""
    size = np.array(size, dtype=np.float64)
    return _CUBE_3x8 * size.reshape(3, 1) / 2


def _transform_coordinates_3d(coordinates, sRT):
    """Exact copy of cutoop's transform_coordinates_3d. [3,N] -> [3,N]."""
    coordinates = np.vstack(
        [coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)]
    )
    new_coordinates = sRT @ coordinates
    return new_coordinates[:3, :] / new_coordinates[3, :]


def _calculate_2d_projections(coordinates_3d, intrinsics_K):
    """Exact copy of cutoop's calculate_2d_projections. Returns [N, 2] int32."""
    projected = intrinsics_K @ coordinates_3d
    projected = projected[:2, :] / projected[2, :]
    projected = projected.transpose()
    return np.array(projected, dtype=np.int32)


def _draw_bboxes(img, projected_bbox_Nx2, transformed_bbox_Nx3, color=None, thickness=1):
    """Exact copy of cutoop's draw_bboxes."""
    if color is None:
        colors = (_bbox_coef * 255 + 255) / 2
        colors = np.clip(colors * 0.82 + np.array([18, 22, 26]), 0, 255)
    else:
        colors = np.tile(np.array(color).reshape(3), [8, 1])

    projected_bbox_Nx2 = np.int32(projected_bbox_Nx2).reshape(-1, 2)
    lines = list(_BBOX_LINES)
    lines.sort(
        reverse=True,
        key=lambda x: (
            (transformed_bbox_Nx3[x[0]] + transformed_bbox_Nx3[x[1]]) ** 2
        ).sum(),
    )
    for i, j in lines:
        img = _draw_seg(
            img,
            projected_bbox_Nx2[i],
            projected_bbox_Nx2[j],
            colors[i],
            colors[j],
            thickness_coef=thickness,
        )
    return img


# ── public API ────────────────────────────────────────────────────────────────

def draw_3d_bbox(img, pose, size, intrinsics, color=None, thickness=1,
                 draw_axes_flag=False, axes_length=0.1, axes_pose=None):
    """Exact equivalent of cutoop's draw_3d_bbox + draw_pose_axes."""
    matK = np.array([
        [intrinsics.fx, 0, intrinsics.cx],
        [0, intrinsics.fy, intrinsics.cy],
        [0, 0, 1],
    ])
    h, w = img.shape[0], img.shape[1]
    scale_2d = np.array([w / intrinsics.width, h / intrinsics.height])

    bbox_3d = _create_3d_bbox(size)                              # [3, 8]
    transformed = _transform_coordinates_3d(bbox_3d, pose)      # [3, 8]
    projected = _calculate_2d_projections(transformed, matK)    # [8, 2]
    projected = np.array(projected * scale_2d[None, :], dtype=np.int32)

    img = _draw_bboxes(img, projected, transformed.T, color=color, thickness=thickness)

    if draw_axes_flag:
        img = draw_pose_axes(img, axes_pose if axes_pose is not None else pose,
                             intrinsics, axes_length, thickness)

    return img


def draw_pose_axes(img, pose, intrinsics, length=0.1, thickness=1):
    """Exact equivalent of cutoop's draw_pose_axes."""
    matK = np.array([
        [intrinsics.fx, 0, intrinsics.cx],
        [0, intrinsics.fy, intrinsics.cy],
        [0, 0, 1],
    ])
    h, w = img.shape[0], img.shape[1]
    scale_2d = np.array([w / intrinsics.width, h / intrinsics.height])

    # print(np.concatenate([np.zeros((3, 1)), np.diag([length, length, length]) / 2], axis=1))
    # print(pose)

    axes_ends = _transform_coordinates_3d(
        np.concatenate([np.zeros((3, 1)), np.diag([length*3, length*3, length*3]) / 2], axis=1),
        pose,
    )  # [3, 4]

    # print(axes_ends)
    # print(axes_ends[:,1]-axes_ends[:,0], axes_ends[:,2]-axes_ends[:,0], axes_ends[:,3]-axes_ends[:,0])
    # print(_calculate_2d_projections(axes_ends, matK))
    origin, ax, ay, az = np.array(
        _calculate_2d_projections(axes_ends, matK) * scale_2d[None, :],
        dtype=np.int32,
    )
    # print(origin, ax, ay, az)
    img = _draw_seg(img, origin, ay, np.array([0, 255, 0]), thickness_coef=thickness)
    img = _draw_seg(img, origin, ax, np.array([0, 0, 255]), thickness_coef=thickness)
    img = _draw_seg(img, origin, az, np.array([255, 0, 0]), thickness_coef=thickness)
    # img = _draw_seg(img, origin, ay, np.array([0, 255, 0]), thickness_coef=thickness)
    return img


def visualize_detections(img, poses, lengths, intrinsics, color=None,
                         thickness=1, draw_axes=True, axes_length=0.1,
                         bbox_poses=None):
    """Draw all detected objects using cutoop-identical visualization."""
    for idx, (pose, length) in enumerate(zip(poses, lengths)):
        bbox_pose = bbox_poses[idx] if bbox_poses is not None else pose
        img = draw_3d_bbox(img, bbox_pose, length, intrinsics, color=color,
                           thickness=thickness, draw_axes_flag=draw_axes,
                           axes_length=axes_length, axes_pose=pose)
    return img

import torch
def draw_kept(batch_sample, keypoint, vis, args):
    # ROI image(s) and inverse affine transform(s)
    roi_rgb = batch_sample.get('roi_rgb_', None)

    # Determine ROI width/height (robust to batched/unbatched tensors)
    if roi_rgb is not None:
        if isinstance(roi_rgb, torch.Tensor):
            # roi_rgb expected shape: [bs, H, W, 3]
            if roi_rgb.dim() == 4:
                output_h, output_w = int(roi_rgb.shape[1]), int(roi_rgb.shape[2])
            elif roi_rgb.dim() == 3:
                output_h, output_w = int(roi_rgb.shape[0]), int(roi_rgb.shape[1])
            else:
                output_h, output_w = args.img_size, args.img_size
        else:
            output_h, output_w = roi_rgb.shape[:2]
    else:
        output_h, output_w = args.img_size, args.img_size

    inv_trans = batch_sample.get('roi_to_img_trans', None)

    # Draw keypoints (both global image and per-ROI visualization)
    if keypoint is not None and batch_sample is not None:
        colors = [
            (255, 0, 0),    # red
            (0, 255, 0),    # green
            (0, 0, 255),    # blue
            (255, 255, 0),  # yellow
            (255, 0, 255),  # magenta
            (0, 255, 255),  # cyan
        ]

        # prepare inverse transform array
        inv_np = None
        try:
            if inv_trans is not None:
                if isinstance(inv_trans, torch.Tensor):
                    inv_np = inv_trans.cpu().numpy()
                else:
                    inv_np = np.asarray(inv_trans)
        except Exception as e:
            print(f"Warning converting roi_to_img_trans: {e}")
            inv_np = None

        # draw on global image using inverse affine (ROI -> image)
        if isinstance(keypoint, list) and inv_np is not None:
            for obj_idx, sample in enumerate(keypoint):
                if sample is None:
                    continue
                if obj_idx >= inv_np.shape[0]:
                    continue
                trans = inv_np[obj_idx]  # expected shape (2, 3)
                for cls_idx, cls_kpts in enumerate(sample):
                    for kp in cls_kpts:
                        try:
                            dst_x = kp['x'] * output_w
                            dst_y = kp['y'] * output_h
                            src = np.dot(trans, np.array([dst_x, dst_y, 1.0], dtype=np.float32))
                            gx, gy = int(round(src[0])), int(round(src[1]))
                            conf = kp.get('conf', 1.0)
                            if conf < 0.05:
                                continue
                            color = colors[cls_idx % len(colors)]
                            cv2.circle(vis, (gx, gy), 4, color, -1)
                            cv2.circle(vis, (gx, gy), 6, (255, 255, 255), 1)
                        except Exception:
                            continue
    return vis
