import os
import sys
import argparse
import cv2
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
_orig_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]
from inference.inference_helper import Flow
sys.argv = _orig_argv
from inference.combined_mask import make_combined_mask
from dataset.infer_loader import get_infer_dataloader
from utils.infer_utils import draw_frame_info, show_frame
from utils.yomni_vis import visualize_detections
from utils.transforms.rotation import matrix_to_euler_angles, euler_angles_to_matrix, eulerZYX_to_matrix
import pyrealsense2 as rs
from networks.dino.dino import DinoLoader
from args import parse_arguments
import copy

ex_T = np.array([[0, -0.707, 0.707], [-1, 0, 0], [0, -0.707, -0.707]], dtype=np.float32)


class RealtimePoseAxes3DPlotter:
    def __init__(self, axis_len=0.05, max_range=0.5):
        self.axis_len = axis_len
        self.max_range = max_range

        plt.ion()
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title('Realtime Object Axes (3D)')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.grid(True, alpha=0.3)

    def _set_equal_limits(self, points_xyz):
        if points_xyz.size == 0:
            self.ax.set_xlim(-self.max_range, self.max_range)
            self.ax.set_ylim(-self.max_range, self.max_range)
            self.ax.set_zlim(-self.max_range, self.max_range)
            return

        mins = points_xyz.min(axis=0)
        maxs = points_xyz.max(axis=0)
        center = (mins + maxs) / 2.0
        extent = np.max(maxs - mins)
        half = max(self.max_range, extent * 0.6)

        self.ax.set_xlim(0.0, 0.8)
        self.ax.set_ylim(center[1] - half, center[1] + half)
        self.ax.set_zlim(-0.8, 0.0)

    def update(self, pose_mats):
        pose_np = np.asarray(pose_mats, dtype=np.float32)
        if pose_np.ndim == 2:
            pose_np = pose_np[None, ...]
        if pose_np.ndim != 3 or pose_np.shape[-2:] != (4, 4):
            return

        self.ax.cla()
        self.ax.set_title('Realtime Object Axes (3D)')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.grid(True, alpha=0.3)

        origins = pose_np[:, :3, 3]
        self.ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], c='k', s=20, label='Origin')
        

        for i in range(pose_np.shape[0]):
            ##
            bb = copy.deepcopy(pose_mats[i])
            bb[:3, :3] = torch.tensor(ex_T) @ torch.tensor(bb[:3, :3])
            bb[:3, 3] = torch.tensor(ex_T) @ torch.tensor(bb[:3, 3])
            aa = pose_to_euler_zyx(bb, degrees=False)
            # angle_x = aa['euler_zyx']
            # adjusted_r = np.eye(3)
            # if angle_x[0] < -np.pi/2:
            #     adjusted_r = eulerZYX_to_matrix(torch.tensor([np.pi,0,0]).unsqueeze(0))[0].numpy()
            # if angle_x[0] > np.pi/2:
            #     adjusted_r = eulerZYX_to_matrix(torch.tensor([-np.pi,0,0]).unsqueeze(0))[0].numpy() 
            ##
            origin = (ex_T @ pose_np[i, :3, 3].transpose()).transpose()
            rot = pose_np[i, :3, :3]
            axes = np.diag([self.axis_len, self.axis_len, self.axis_len])
            # x_axis = adjusted_r @ ex_T @ rot @ axes[:, 0]
            # y_axis = adjusted_r @ ex_T @ rot @ axes[:, 1]
            # z_axis = adjusted_r @ ex_T @ rot @ axes[:, 2]

            x_axis = ex_T @ rot @ axes[:, 0]
            y_axis = ex_T @ rot @ axes[:, 1]
            z_axis = ex_T @ rot @ axes[:, 2]

            self.ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color='r', linewidth=1.5)
            self.ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color='g', linewidth=1.5)
            self.ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color='b', linewidth=1.5)
            self.ax.text(origin[0], origin[1], origin[2], f'{i}', fontsize=8, color='k')

            self.ax.quiver(0, 0, -0.5, origin[0], origin[1], origin[2]+0.5, color='k', linewidth=0.5)

            self.ax.quiver(0, 0, -0.5, 0.2, 0, 0, color='r', linewidth=2.0)
            self.ax.quiver(0, 0, -0.5, 0, 0.2, 0, color='g', linewidth=2.0)
            self.ax.quiver(0, 0, -0.5, 0, 0, 0.2, color='b', linewidth=2.0)
            self.ax.text(0, 0, -0.5, 'axes', fontsize=8, color='k')

        self._set_equal_limits(origins)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
        
    def close(self):
        try:
            plt.ioff()
            plt.close(self.fig)
        except Exception:
            pass

def calibrate(pose_mats):
        pose_np = np.asarray(copy.deepcopy(pose_mats), dtype=np.float32)
        for i in range(pose_np.shape[0]):
            bb = copy.deepcopy(pose_np[i])
            bb[:3, :3] = ex_T @ bb[:3, :3]
            aa = pose_to_euler_zyx(torch.tensor(bb), degrees=False)
            angle_x = aa['euler_zyx']
            adjusted_r = np.eye(3)
            if angle_x[0] < -np.pi/2:
                adjusted_r = eulerZYX_to_matrix(torch.tensor([np.pi,0,0]).unsqueeze(0))[0].numpy()
            if angle_x[0] > np.pi/2:
                adjusted_r = eulerZYX_to_matrix(torch.tensor([-np.pi,0,0]).unsqueeze(0))[0].numpy() 
            pose_np[i, :3, :3] = ex_T.transpose() @ adjusted_r @ ex_T @ pose_np[i, :3, :3]
        return torch.tensor(pose_np)


def pose_to_euler_zyx(pose, degrees=False):
    """Convert pose matrix/matrices to Euler ZYX + translation.

    Args:
        pose: torch.Tensor or np.ndarray with shape [4, 4] or [N, 4, 4].
        degrees: If True, return Euler angles in degrees; otherwise radians.

    Returns:
        dict with keys:
            - "euler_zyx": [3] or [N, 3]
            - "translation": [3] or [N, 3]
    """
    is_numpy = isinstance(pose, np.ndarray)
    pose_t = torch.as_tensor(pose)

    if pose_t.ndim == 2:
        pose_t = pose_t.unsqueeze(0)
        squeeze_out = True
    elif pose_t.ndim == 3:
        squeeze_out = False
    else:
        raise ValueError(f"pose must have shape [4,4] or [N,4,4], got {tuple(pose_t.shape)}")

    if pose_t.shape[-2:] != (4, 4):
        raise ValueError(f"pose must end with shape [4,4], got {tuple(pose_t.shape)}")

    rot = pose_t[:, :3, :3].to(torch.float32)
    trans = pose_t[:, :3, 3].to(torch.float32)
    euler_zyx = matrix_to_euler_angles(rot, 'ZYX')
    

    if degrees:
        euler_zyx = torch.rad2deg(euler_zyx)

    if squeeze_out:
        euler_zyx = euler_zyx[0]
        trans = trans[0]

    if is_numpy:
        return {
            "euler_zyx": euler_zyx.cpu().numpy(),
            "translation": trans.cpu().numpy(),
        }

    return {
        "euler_zyx": euler_zyx,
        "translation": trans,
    }


def euler_zyx_to_pose(euler_zyx, translation, degrees=False):
    """Convert Euler ZYX + translation back to original pose matrix format.

    Args:
        euler_zyx: torch.Tensor or np.ndarray with shape [3] or [N, 3].
        translation: torch.Tensor or np.ndarray with shape [3] or [N, 3].
        degrees: If True, euler_zyx is interpreted in degrees; otherwise radians.

    Returns:
        pose matrix/matrices with shape [4, 4] or [N, 4, 4], matching input type.
    """
    input_is_numpy = isinstance(euler_zyx, np.ndarray) and isinstance(translation, np.ndarray)

    euler_t = torch.as_tensor(euler_zyx, dtype=torch.float32)
    trans_t = torch.as_tensor(translation, dtype=torch.float32, device=euler_t.device)

    if euler_t.ndim == 1:
        euler_t = euler_t.unsqueeze(0)
        squeeze_out = True
    elif euler_t.ndim == 2:
        squeeze_out = False
    else:
        raise ValueError(f"euler_zyx must have shape [3] or [N,3], got {tuple(euler_t.shape)}")

    if trans_t.ndim == 1:
        trans_t = trans_t.unsqueeze(0)
    elif trans_t.ndim != 2:
        raise ValueError(f"translation must have shape [3] or [N,3], got {tuple(trans_t.shape)}")

    if euler_t.shape[-1] != 3 or trans_t.shape[-1] != 3:
        raise ValueError("euler_zyx and translation must both have last dimension 3")

    if euler_t.shape[0] != trans_t.shape[0]:
        raise ValueError(
            f"batch size mismatch: euler_zyx has {euler_t.shape[0]}, translation has {trans_t.shape[0]}"
        )

    if degrees:
        euler_t = torch.deg2rad(euler_t)

    rot = eulerZYX_to_matrix(euler_t)  # euler_t: [..., 0]=Z(yaw), [..., 1]=Y(pitch), [..., 2]=X(roll)
    pose = torch.eye(4, dtype=rot.dtype, device=rot.device).unsqueeze(0).repeat(rot.shape[0], 1, 1)
    pose[:, :3, :3] = rot
    pose[:, :3, 3] = trans_t

    if squeeze_out:
        pose = pose[0]

    if input_is_numpy:
        return pose.cpu().numpy()

    return pose


def realsense_pipeline(w, h, fps):
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f'RealSense initialized (depth scale = {depth_scale} m/unit)')
    return pipeline, align, depth_scale

def wait_frames(pipeline):
    try:
        frames = pipeline.wait_for_frames(timeout_ms=1000)
        return frames
    except Exception as e:
        print(f'RealSense frame timeout or error: {e}. Retrying...')
        pipeline.stop()
        time.sleep(1)

def release_realsense(pipeline=None, align=None):
    try:
        if pipeline is not None:
            pipeline.stop()
            
    except Exception as e:
        print(f'Error stopping RealSense pipeline: {e}')
    finally:
        del pipeline
        del align
        time.sleep(0.5)

def process_frame(frame, depth_raw, flow, yolo, args, writer, frame_idx, depth_scale, dino_loader,
                  axes_plotter=None):
    # run YOLO
    results = yolo.track(frame, conf = 0.25, iou=0.5, persist=True, tracker="bytetrack.yaml", verbose=False)
    boxes = results[0].boxes
    masks = results[0].masks
    # show YOLO detections
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO', annotated_frame)
    # cv2.waitKey(1)
    # return None, None, 0.0

    # check for no detections
    if masks is None or len(masks.data) == 0:
        vis = frame.copy()
        num_detections = 0
        draw_frame_info(vis, frame_idx, 0.00, num_detections)
        show_frame(vis, writer, args)
        return None, None, 0.0

    # Extract track IDs for pose persistence
    box_ids = None
    if boxes is not None and hasattr(boxes, 'id') and boxes.id is not None:
        box_ids = [int(id_val) for id_val in boxes.id.cpu().numpy()]

    combined_mask, obj_ids = make_combined_mask(frame.shape[0], frame.shape[1], masks, box_ids)
    

    if obj_ids is None or (len(np.unique(combined_mask)) != len(obj_ids)):
        vis = frame.copy()
        num_detections = len(masks.data) if masks is not None else 0
        draw_frame_info(vis, frame_idx, None, num_detections)
        show_frame(vis, writer, args)
        return None, None, 0.0

    frame_data = {
        "color": frame,
        "depth": (depth_raw.astype(np.float32) * depth_scale * 1000).astype(np.uint16) if depth_raw is not None else None,
        "mask": combined_mask
    }

    try:
        data = get_infer_dataloader(frame_data, args, mode='rs')
    except Exception as e:
        print(f'Error in get_infer_dataloader: {e}')
        return None, None, 0.0

    obj_ids = list(filter(lambda row: row != [0,0] and row != [255,255], obj_ids))
    if len(obj_ids) == 0 or data is None:
        return None, None, 0.0

    t0 = time.time()
    if args.tracking:
        pose, length = flow.inference(data, dino_loader = dino_loader, obj_ids=obj_ids, frame_idx=frame_idx, enable_tracking=True)
    else:
        pose, length = flow.inference(data, dino_loader = dino_loader, obj_ids=obj_ids, frame_idx=frame_idx, enable_tracking=False)
    t1 = time.time()

    pose[0] = calibrate(pose[0])

    # pose_adjusted = copy.deepcopy(pose)
    # if axes_plotter is not None:
        # axes_plotter.update(pose[0])
        # pose[0] = axes_plotter.calibrate(pose[0])
        # axes_plotter.update(pose[0])

    
    vis = frame.copy()
    valid_output = (
        isinstance(pose, (list, tuple)) and isinstance(length, (list, tuple))
        and len(pose) > 0 and len(length) > 0
        and pose[0] is not None and length[0] is not None
    )

    if valid_output:
        all_final_pose = pose[0].to(torch.float32).cpu().numpy()
        all_final_length = length[0].to(torch.float32).cpu().numpy()
        vis = visualize_detections(vis, all_final_pose, all_final_length, data.cam_intrinsics, thickness=1)
        if axes_plotter is not None:
            axes_plotter.update(all_final_pose)

    num_detections = len(masks.data) if masks is not None else 0
    draw_frame_info(vis, frame_idx, t1-t0, num_detections)
    show_frame(vis, writer, args)
    return pose, length, t1 - t0

def main():
    args = parse_arguments()
    VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
    VIDEO_FPS = 30
    rs_pipeline = None
    rs_align = None
    depth_scale = None

    device = args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu'
    dino_loader = DinoLoader(model_name='dinov2_vits14', device=args.device)

    flow = Flow(args)
    yolo = YOLO("results/ckpts/YOLO/mixed.pt")

    rs_pipeline, rs_align, depth_scale = realsense_pipeline(VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS)

    writer = None
    frame_idx = 0
    frames = None
    # axes_plotter = RealtimePoseAxes3DPlotter(axis_len=0.1, max_range=0.5)
    # axes_plotter = None

    dino_loader = DinoLoader(model_name='dinov2_vits14', device=args.device)

    try:
        while True:
            frames = wait_frames(rs_pipeline)
            if frames is None:
                rs_pipeline, rs_align, depth_scale = realsense_pipeline(VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS)
                continue
            aligned = rs_align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())

            pose, length, infer_time = process_frame(
                frame, depth_raw, flow, yolo, args, writer, frame_idx, depth_scale, dino_loader)
            frame_idx += 1

    except KeyboardInterrupt:
        print('Interrupted by user')

    finally:
        if args.realsense and rs_pipeline is not None:
            try:
                release_realsense(rs_pipeline, rs_align)
                rs_pipeline.stop()
            except Exception:
                pass
        if writer is not None:
            writer.release()
        # if axes_plotter is not None:
        #     axes_plotter.close()
        if args.show:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()