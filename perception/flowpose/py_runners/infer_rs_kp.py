import os
import sys
import argparse
import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
_orig_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]
from inference.inference_helper_kp import Flow
sys.argv = _orig_argv
from networks.dino.dino import DinoLoader
from inference.combined_mask import make_combined_mask
from dataset.infer_loader import get_infer_dataloader
from utils.infer_utils import draw_frame_info, show_frame
from utils.yomni_vis import visualize_detections
from args import parse_arguments
import pyrealsense2 as rs
import copy
from utils.transforms.rotation import matrix_to_euler_angles, eulerZYX_to_matrix

ex_T = np.array([[0, -0.707, 0.707], [-1, 0, 0], [0, -0.707, -0.707]], dtype=np.float32)

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

def process_frame(frame, depth_raw, dino_loader, flow, yolo, args, writer, frame_idx, depth_scale):
    # run YOLO
    results = yolo.track(frame, conf=0.3, iou=0.5, persist=True, tracker="bytetrack.yaml", verbose=False)
    boxes = results[0].boxes
    masks = results[0].masks
    
    # show YOLO detections
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO', annotated_frame)

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
        data = get_infer_dataloader(frame_data, args)
    except Exception as e:
        print(f'Error in get_infer_dataloader: {e}')
        return None, None, 0.0

    obj_ids = list(filter(lambda row: row != [0,0] and row != [255,255], obj_ids))
    if len(obj_ids) == 0 or data is None:
        return None, None, 0.0

    t0 = time.time()
    pose, length, keypoint, batch_sample = flow.inference(data, dino_loader, obj_ids=obj_ids, frame_idx=frame_idx, enable_tracking=True, mode='full')
    t1 = time.time()

    if len(pose) > 0 and pose[0] is not None:
        pose[0] = calibrate(pose[0]) if pose is not None else None
    else:
        return None, None, 0.0

    if keypoint is None:
        print('Keypoint result: None')
    elif isinstance(keypoint, torch.Tensor):
        print(f'Keypoint result shape: {keypoint.shape}')
    elif isinstance(keypoint, list):
        # keypoint: List[batch] where each batch item is List[classes] of keypoint dicts
        batch_size = len(keypoint)
        per_sample_counts = []
        for sample in keypoint:
            if isinstance(sample, list):
                per_sample_counts.append([len(cls) for cls in sample])
            else:
                per_sample_counts.append(len(sample))
        # print(f'Keypoint result: list, batch={batch_size}, counts={per_sample_counts}')
    else:
        print(f'Keypoint result type: {type(keypoint)}')

    # Visualize keypoints on both ROI and global image (use kp_val style)
    vis = frame.copy()
    if keypoint is not None and batch_sample is not None:
        roi_rgb = batch_sample.get('roi_rgb_', None)
        inv_trans = batch_sample.get('roi_to_img_trans', None)

        # Determine ROI width/height (robust to batched/unbatched tensors)
        if roi_rgb is not None:
            if isinstance(roi_rgb, torch.Tensor):
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

        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

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
                            dst_x = kp.get('x', None)
                            dst_y = kp.get('y', None)
                            if dst_x is None or dst_y is None:
                                continue
                            dst_x = dst_x * output_w
                            dst_y = dst_y * output_h
                            src = np.dot(trans, np.array([dst_x, dst_y, 1.0], dtype=np.float32))
                            gx, gy = int(round(src[0])), int(round(src[1]))
                            conf = kp.get('conf', 1.0)
                            if conf < 0.05:
                                continue
                            color = colors[cls_idx % len(colors)]
                            # cv2.circle(vis, (gx, gy), 4, color, -1)
                            cv2.circle(vis, (gx, gy), 6, (255, 255, 255), 1)
                        except Exception:
                            continue

        # show per-ROI keypoints in small windows (useful for debugging)
        # if roi_rgb is not None:
        #     try:
        #         if isinstance(roi_rgb, torch.Tensor):
        #             roi_np = roi_rgb.cpu().numpy()
        #         else:
        #             roi_np = np.asarray(roi_rgb)
        #     except Exception as e:
        #         roi_np = None
        #     # roi visualization with keypoints
        #     if roi_np is not None:
        #         # roi_np expected shape [bs, H, W, 3]
        #         for i in range(roi_np.shape[0]):
        #             roi_vis = roi_np[i].copy()
        #             if isinstance(keypoint, list) and i < len(keypoint):
        #                 for cls_list in keypoint[i]:
        #                     for kp in cls_list:
        #                         x_norm = kp.get('x', None)
        #                         y_norm = kp.get('y', None)
        #                         if x_norm is None or y_norm is None:
        #                             continue
        #                         u = int(round(x_norm * (roi_np.shape[2] - 1)))
        #                         v = int(round(y_norm * (roi_np.shape[1] - 1)))
        #                         cv2.circle(roi_vis, (u, v), 3, (0, 255, 0), -1)
        #             win_name = f'ROI_{i}'
        #             cv2.imshow(win_name, roi_vis)
    ###
    # vis = frame.copy()
    valid_output = (
        isinstance(pose, (list, tuple)) and isinstance(length, (list, tuple))
        and len(pose) > 0 and len(length) > 0
        and pose[0] is not None and length[0] is not None
    )

    if valid_output:
        all_final_pose = pose[0].to(torch.float32).cpu().numpy()
        all_final_length = length[0].to(torch.float32).cpu().numpy()
        vis = visualize_detections(vis, all_final_pose, all_final_length, data.cam_intrinsics, thickness=1)

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

    dino_loader = DinoLoader(model_name='dinov2_vits14', device=device)
    flow = Flow(args)
    yolo = YOLO("results/ckpts/YOLO/mixed.pt")

    rs_pipeline, rs_align, depth_scale = realsense_pipeline(VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS)

    writer = None
    frame_idx = 0
    frames = None

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

            pose, length, infer_time = process_frame(frame, depth_raw, dino_loader, flow, yolo, args, writer, frame_idx, depth_scale)
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
        if args.show:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()