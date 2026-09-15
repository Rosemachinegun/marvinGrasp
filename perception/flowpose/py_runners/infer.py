import os 
import sys
import glob
import time
import torch
import cv2
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
_orig_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]
from inference.inference_helper import Flow
sys.argv = _orig_argv

from dataset.infer_loader import get_infer_dataloader
from utils.infer_utils import draw_frame_info, show_frame
from utils.yomni_vis import visualize_detections
from args import parse_arguments
from networks.dino.dino import DinoLoader

def process_frame(frame, flow, args, frame_idx, dino_loader):
    batch_sample = frame.get_objects()
    labels = batch_sample['labels']
    vis = frame._color.copy()
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    obj_ids = [(int(lbl), int(i)) for i, lbl in enumerate(labels)]

    t0 = time.time()
    pose, length = flow.inference(frame, dino_loader=dino_loader, obj_ids=obj_ids, frame_idx=frame_idx, enable_tracking=args.tracking)
    t1 = time.time()
    inference_time = t1 - t0
    num_detections = len(pose)

    valid_output = (
        isinstance(pose, (list, tuple)) and isinstance(length, (list, tuple))
        and len(pose) > 0 and len(length) > 0
        and pose[0] is not None and length[0] is not None
    )

    if valid_output:
        all_final_pose = pose[0].to(torch.float32).cpu().numpy()
        all_final_length = length[0].to(torch.float32).cpu().numpy()
        vis = visualize_detections(
            vis,
            all_final_pose,
            all_final_length,
            frame.cam_intrinsics,
            thickness=1,
            draw_axes=True,
            axes_length=0.1,
        )

    return vis, inference_time, num_detections

def main():
    args = parse_arguments()
    
    flow = Flow(args)
    writer = None
    frame_idx = 0
    rgb = sorted(glob.glob(args.data_path + '/*_color.png'))
    dino_loader = DinoLoader(model_name='dinov2_vits14', device=args.device)

    for i, full_path in enumerate(tqdm(rgb)):
        data_prefix = full_path.replace('color.png', '')
        frame = get_infer_dataloader(data_prefix, args, mode=args.data_mode)

        vis, inference_time, num_detections = process_frame(frame, flow, args, frame_idx, dino_loader)
        draw_frame_info(vis, frame_idx, inference_time, num_detections)
        show_frame(vis, writer, args, waitkey=True)
        frame_idx += 1


if __name__ == "__main__":
    main()