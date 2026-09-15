import os
import sys
import argparse
import cv2
import numpy as np
import torch
import time
import logging
import open3d as o3d

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import pyrealsense2 as rs

code_dir = os.path.dirname(os.path.realpath(__file__))
# Add root to sys.path
sys.path.append(os.path.dirname(code_dir))
# Add Fast-FoundationStereo to sys.path
sys.path.append(os.path.join(os.path.dirname(code_dir), 'Fast-FoundationStereo'))

from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import set_logging_format, set_seed

from ultralytics import YOLO

_orig_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]
from inference.inference_helper import Flow
sys.argv = _orig_argv
from inference.combined_mask import make_combined_mask
from dataset.infer_loader import get_infer_dataloader
from utils.infer_utils import draw_frame_info, show_frame
from utils.yomni_vis import visualize_detections
from args import parse_arguments
from networks.dino.dino import DinoLoader

# ── Non-blocking Open3D point-cloud viewer ──────────────────────────────────
_pcl_vis = None
_pcl_geoms = []

COLORS_TAB10 = [
    [0.122, 0.467, 0.706],  # blue
    [1.000, 0.498, 0.055],  # orange
    [0.173, 0.627, 0.173],  # green
    [0.839, 0.153, 0.157],  # red
    [0.580, 0.404, 0.741],  # purple
    [0.549, 0.337, 0.294],  # brown
    [0.890, 0.467, 0.761],  # pink
    [0.498, 0.498, 0.498],  # gray
    [0.737, 0.741, 0.133],  # olive
    [0.090, 0.745, 0.812],  # cyan
]

def visualize_segmented_pcl(data):
    """Show per-object point clouds with projected RGB in an Open3D window (non-blocking)."""
    global _pcl_vis, _pcl_geoms

    batch_sample = data.get_objects()
    if batch_sample is None:
        return

    pts = batch_sample['pts'].cpu().numpy()           # [N_obj, 1024, 3]
    roi_xs = batch_sample['roi_xs'].cpu().numpy()     # [N_obj, 1024] — row indices
    roi_ys = batch_sample['roi_ys'].cpu().numpy()     # [N_obj, 1024] — col indices

    # Reverse ImageNet normalization on roi_rgb [N_obj, 3, H, W] to get uint8 images
    roi_rgb = batch_sample['roi_rgb'].cpu().numpy()   # [N_obj, 3, H, W] float32 normalized
    _mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    _std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    roi_rgb_denorm = np.clip((roi_rgb * _std + _mean) * 255, 0, 255).astype(np.uint8)
    roi_rgb_denorm = roi_rgb_denorm.transpose(0, 2, 3, 1)  # [N_obj, H, W, 3]

    # First call: create the window and zoom in
    if _pcl_vis is None:
        _pcl_vis = o3d.visualization.Visualizer()
        _pcl_vis.create_window('Segmented Point Clouds', width=640, height=480)
        opt = _pcl_vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.point_size = 3.0

    # Remove previous geometries
    for g in _pcl_geoms:
        _pcl_vis.remove_geometry(g, reset_bounding_box=False)
    _pcl_geoms.clear()

    reset_cam = (len(_pcl_geoms) == 0)
    for i in range(pts.shape[0]):
        pcd = o3d.geometry.PointCloud()
        # Flip Y and Z to convert from camera coords (Y-down, Z-forward)
        # to Open3D coords (Y-up, Z-backward) so the view is right-side-up
        flipped = pts[i].copy()
        flipped[:, 1] *= -1
        flipped[:, 2] *= -1
        pcd.points = o3d.utility.Vector3dVector(flipped)
        # Project RGB from the cropped ROI image onto each point
        colors = roi_rgb_denorm[i][roi_xs[i], roi_ys[i]]  # [1024, 3] uint8
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        _pcl_vis.add_geometry(pcd, reset_bounding_box=reset_cam)
        _pcl_geoms.append(pcd)
        reset_cam = False

    # Zoom in every frame
    ctr = _pcl_vis.get_view_control()
    ctr.set_zoom(0.3)

    _pcl_vis.poll_events()
    _pcl_vis.update_renderer()


def process_frame(frame, depth_raw, dino_loader, flow, yolo, args, writer, frame_idx, depth_scale):
    # run YOLO
    results = yolo.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
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

    # depth_raw here is expected to be in meters, and we use depth_scale=1.0 
    # so we multiply by 1000 to get uint16 mm depth needed for dataloader
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

    # Visualize segmented point clouds
    try:
        visualize_segmented_pcl(data)
    except Exception as e:
        print(f'PCL vis error: {e}')

    t0 = time.time()
    if args.tracking:
        pose, length = flow.inference(data, dino_loader, obj_ids=obj_ids, frame_idx=frame_idx, enable_tracking=True)
    else:
        pose, length = flow.inference(data, dino_loader, obj_ids=obj_ids, frame_idx=frame_idx, enable_tracking=False)
    t1 = time.time()
    # print(f'Pose inference time for frame {frame_idx}: {t1 - t0:.3f} seconds')

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

    num_detections = len(masks.data) if masks is not None else 0
    draw_frame_info(vis, frame_idx, t1-t0, num_detections)
    show_frame(vis, writer, args)
    return pose, length, t1 - t0

def main():
    args = parse_arguments()
    VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
    VIDEO_FPS = 30
    
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    dino_loader = DinoLoader(model_name='dinov2_vits14', device='cuda')

    # 1. Initialize Flow and YOLO
    flow = Flow(args)
    yolo = YOLO("results/ckpts/YOLO/best.pt")

    # 2. Initialize FoundationStereo Model
    stereo_ckpt = os.path.join(code_dir, '../Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth')
    cfg_path = os.path.join(os.path.dirname(stereo_ckpt), 'cfg.yaml')
    cfg = OmegaConf.load(cfg_path)
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    
    # Merge stereo args with parsed args if necessary
    fs_args = OmegaConf.create(cfg)
    # Give some defaults if not in config
    valid_iters = getattr(fs_args, 'valid_iters', 8)
    max_disp = getattr(fs_args, 'max_disp', 256)
    
    print(f"Loading FoundationStereo model from {stereo_ckpt}")
    stereo_model = torch.load(stereo_ckpt, map_location='cpu', weights_only=False)
    stereo_model.args.valid_iters = valid_iters
    stereo_model.args.max_disp = max_disp
    stereo_model.cuda()
    stereo_model.eval()

    # 3. Initialize RealSense for Stereo & Color
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, VIDEO_WIDTH, VIDEO_HEIGHT, rs.format.y8, VIDEO_FPS)  # Left IR
    config.enable_stream(rs.stream.infrared, 2, VIDEO_WIDTH, VIDEO_HEIGHT, rs.format.y8, VIDEO_FPS)  # Right IR
    config.enable_stream(rs.stream.color, VIDEO_WIDTH, VIDEO_HEIGHT, rs.format.bgr8, VIDEO_FPS)       # Color
    profile = pipeline.start(config)

    # Disable IR emitter for clean passive IR imaging
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 0)

    # Intrinsics and Extrinsics
    left_stream = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    right_stream = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    
    left_intr = left_stream.get_intrinsics()
    color_intr = color_stream.get_intrinsics()
    extr = left_stream.get_extrinsics_to(right_stream)
    ir_to_color_extr = left_stream.get_extrinsics_to(color_stream)
    
    baseline = abs(extr.translation[0])

    R_ext = np.array(ir_to_color_extr.rotation).reshape(3, 3).T
    T_ext = np.array(ir_to_color_extr.translation)

    # Assuming scale = 1.0 (from Fast-FoundationStereo's args.scale)
    scale = 1.0
    K = np.array([
        [left_intr.fx, 0,            left_intr.ppx],
        [0,            left_intr.fy, left_intr.ppy],
        [0,            0,            1             ],
    ], dtype=np.float32)

    writer = None
    frame_idx = 0

    print("Pipeline started. Press ESC to quit.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            left_frame_rs = frames.get_infrared_frame(1)
            right_frame_rs = frames.get_infrared_frame(2)
            color_frame_rs = frames.get_color_frame()
            if not left_frame_rs or not right_frame_rs or not color_frame_rs:
                continue
            
            color_img = np.asanyarray(color_frame_rs.get_data())  # raw BGR

            # Convert to numpy (grayscale uint8)
            img0_gray = np.asanyarray(left_frame_rs.get_data())
            img1_gray = np.asanyarray(right_frame_rs.get_data())

            # Convert grayscale -> BGR
            img0 = cv2.cvtColor(img0_gray, cv2.COLOR_GRAY2BGR)
            img1 = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)

            H, W = img0.shape[:2]

            img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
            img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
            padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
            img0_t, img1_t = padder.pad(img0_t, img1_t)

            t0 = time.time()
            with torch.amp.autocast('cuda', enabled=True):
                disp = stereo_model.forward(img0_t, img1_t, iters=valid_iters, test_mode=True)
            t1 = time.time()
            # print(f'Stereo inference time for frame {frame_idx}: {t1 - t0:.3f} seconds')

            disp = padder.unpad(disp.float())
            disp = disp.data.cpu().numpy().reshape(H, W)

            # Remove invisible (non-overlapping) pixels from disparity
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            us_right = xx - disp
            disp[us_right < 0] = np.inf

            # Convert disparity to depth (meters)
            depth_ir = K[0, 0] * baseline / disp
            
            # Align depth to Color camera geometry
            Z = depth_ir
            valid = (Z > 0) & np.isfinite(Z)
            y_ir, x_ir = np.nonzero(valid)
            z_ir = Z[valid]
            
            # Undistort IR normalized coordinates before unprojection
            x_n = (x_ir - K[0, 2]) / K[0, 0]
            y_n = (y_ir - K[1, 2]) / K[1, 1]
            ic = left_intr.coeffs
            if left_intr.model == rs.distortion.inverse_brown_conrady:
                r2 = x_n**2 + y_n**2
                f = 1.0 + ic[0]*r2 + ic[1]*r2**2 + ic[4]*r2**3
                ux = x_n*f + 2*ic[2]*x_n*y_n + ic[3]*(r2 + 2*x_n**2)
                uy = y_n*f + ic[2]*(r2 + 2*y_n**2) + 2*ic[3]*x_n*y_n
                x_n, y_n = ux, uy
            elif left_intr.model in (rs.distortion.brown_conrady, rs.distortion.modified_brown_conrady):
                xd, yd = x_n.copy(), y_n.copy()
                for _ in range(10):
                    r2 = x_n**2 + y_n**2
                    f = 1.0 + ic[0]*r2 + ic[1]*r2**2 + ic[4]*r2**3
                    dx = 2*ic[2]*x_n*y_n + ic[3]*(r2 + 2*x_n**2)
                    dy = ic[2]*(r2 + 2*y_n**2) + 2*ic[3]*x_n*y_n
                    x_n = (xd - dx) / f
                    y_n = (yd - dy) / f
            X_ir = x_n * z_ir
            Y_ir = y_n * z_ir
            P_ir = np.stack((X_ir, Y_ir, z_ir), axis=0)
            
            P_color = R_ext @ P_ir + T_ext[:, None]
            X_c, Y_c, Z_c = P_color[0], P_color[1], P_color[2]
            
            # Apply color camera distortion for accurate projection
            x_n = X_c / Z_c
            y_n = Y_c / Z_c
            cc = color_intr.coeffs
            r2 = x_n**2 + y_n**2
            if color_intr.model in (rs.distortion.modified_brown_conrady, rs.distortion.inverse_brown_conrady):
                f = 1.0 + cc[0]*r2 + cc[1]*r2**2 + cc[4]*r2**3
                x_f = x_n * f
                y_f = y_n * f
                xd = x_f + 2*cc[2]*x_f*y_f + cc[3]*(r2 + 2*x_f**2)
                yd = y_f + cc[2]*(r2 + 2*y_f**2) + 2*cc[3]*x_f*y_f
            elif color_intr.model == rs.distortion.brown_conrady:
                f = 1.0 + cc[0]*r2 + cc[1]*r2**2 + cc[4]*r2**3
                xd = x_n*f + 2*cc[2]*x_n*y_n + cc[3]*(r2 + 2*x_n**2)
                yd = y_n*f + cc[2]*(r2 + 2*y_n**2) + 2*cc[3]*x_n*y_n
            else:
                xd = x_n
                yd = y_n
            x_c = xd * color_intr.fx + color_intr.ppx
            y_c = yd * color_intr.fy + color_intr.ppy
            
            x_c = np.round(x_c).astype(int)
            y_c = np.round(y_c).astype(int)
            
            mask = (x_c >= 0) & (x_c < color_intr.width) & (y_c >= 0) & (y_c < color_intr.height) & (Z_c > 0)
            x_c, y_c, Z_c = x_c[mask], y_c[mask], Z_c[mask]
            
            depth_aligned = np.zeros((color_intr.height, color_intr.width), dtype=np.float32)
            order = np.argsort(Z_c)[::-1]
            depth_aligned[y_c[order], x_c[order]] = Z_c[order]
            
            # Run existing FlowPose infer via process_frame.
            # Depth comes out in meters. We use a depth_scale of 1.0 here because
            # process_frame multiplies depth * depth_scale * 1000 to get uint16 mm.
            # So depth (meters) * 1.0 * 1000 = depth in mm.
            pose, length, infer_time = process_frame(
                frame=color_img, 
                depth_raw=depth_aligned, 
                dino_loader=dino_loader, 
                flow=flow, 
                yolo=yolo, 
                args=args, 
                writer=writer, 
                frame_idx=frame_idx, 
                depth_scale=1.0
            )
            total_infer_time = infer_time + (t1 - t0)
            # print(f'Total inference time for frame {frame_idx}: {total_infer_time:.3f} seconds')

            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

    except KeyboardInterrupt:
        print('Interrupted by user')

    finally:
        pipeline.stop()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        if _pcl_vis is not None:
            _pcl_vis.destroy_window()

if __name__ == '__main__':
    main()
