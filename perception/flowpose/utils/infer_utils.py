import cv2

def draw_frame_info(vis, frame_idx, inference_time, num_detections):
    if inference_time is None:
        info = f'Frame {frame_idx}  Inference: SKIPPED (mask mismatch)  Detections: {num_detections}'
    else:
        info = f'Frame {frame_idx}  Inference: {inference_time:.2f}s  Detections: {num_detections}'
    cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return vis

def show_frame(vis, writer, args, waitkey=False):
    if args.show:
        cv2.imshow('flow', vis)
        if waitkey:
            key = cv2.waitKey(0)  # wait for keypress
            if key & 0xFF == ord('q'):
                raise KeyboardInterrupt()
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt()
    if writer is not None:
        writer.write(vis)
    return True