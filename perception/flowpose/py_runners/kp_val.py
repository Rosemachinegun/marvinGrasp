import os
import sys
import cv2
import glob
import numpy as np
import torch
import time
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
_orig_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]
from inference.inference_helper_kp import Flow
sys.argv = _orig_argv
from networks.dino.dino import DinoLoader
from dataset.infer_loader import get_infer_dataloader
from utils.infer_utils import draw_frame_info, show_frame
from utils.yomni_vis import visualize_detections
from args import parse_arguments

def process_frame(frame, dino_loader, flow, args, writer, frame_idx, depth_scale, mode = 'rgb'):
    # batch_sample = frame.get_objects(mode=mode)
    # labels = batch_sample['labels']
    # vis = frame._color.copy()

    # obj_ids = [(int(lbl), int(i)) for i, lbl in enumerate(labels)]
    
    t0 = time.time()
    pose, length, keypoint, batch_sample = flow.inference(frame, dino_loader, obj_ids=None, frame_idx=frame_idx, enable_tracking=False, mode=mode)
    t1 = time.time()    

    vis = frame._color.copy()
    
    
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

        # show per-ROI keypoints in small windows (useful for debugging)
        if roi_rgb is not None:
            try:
                if isinstance(roi_rgb, torch.Tensor):
                    roi_np = roi_rgb.cpu().numpy()
                else:
                    roi_np = np.asarray(roi_rgb)
            except Exception as e:
                roi_np = None

            if roi_np is not None:
                # roi_np expected shape [bs, H, W, 3]
                for i in range(roi_np.shape[0]):
                    roi_vis = roi_np[i].copy()
                    if isinstance(keypoint, list) and i < len(keypoint):
                        for cls_list in keypoint[i]:
                            for kp in cls_list:
                                x_norm = kp.get('x', None)
                                y_norm = kp.get('y', None)
                                if x_norm is None or y_norm is None:
                                    continue
                                u = int(round(x_norm * (roi_np.shape[2] - 1)))
                                v = int(round(y_norm * (roi_np.shape[1] - 1)))
                                cv2.circle(roi_vis, (u, v), 3, (0, 255, 0), -1)
                    win_name = f'ROI_{i}'
                    cv2.imshow(win_name, roi_vis)

    valid_output = (
        isinstance(pose, (list, tuple)) and isinstance(length, (list, tuple))
        and len(pose) > 0 and len(length) > 0
        and pose[0] is not None and length[0] is not None
    )

    if valid_output:
        all_final_pose = pose[0].to(torch.float32).cpu().numpy()
        all_final_length = length[0].to(torch.float32).cpu().numpy()
        vis = visualize_detections(vis, all_final_pose, all_final_length, frame.cam_intrinsics, color=(0, 255, 0), thickness=2, alpha=0.1)
        num_detections = len(pose)
        draw_frame_info(vis, frame_idx, t1-t0, num_detections)
        show_frame(vis, writer, args, waitkey=True)
    else:
        num_detections = 0
    cv2.waitKey(0)
    return pose, length, num_detections

def main():
    args = parse_arguments()
    device = args.device if torch.cuda.is_available() else 'cpu'

    dino_loader = DinoLoader(model_name='dinov2_vits14', device=device)
    flow = Flow(args)

    writer = None
    frame_idx = 0

    rgb = sorted(glob.glob(args.data_path + '/*_color.png'))

    try:
        for i, full_path in enumerate(tqdm(rgb)):
            data_prefix = full_path.replace('color.png', '')
            frame = get_infer_dataloader(data_prefix, args)

            vis, inference_time, num_detections = process_frame(frame, dino_loader, flow, args, writer, frame_idx, None)
            frame_idx += 1

    except Exception as e:
        print(f"Error processing frame {frame_idx}: {e}")

if __name__ == "__main__":
    main()