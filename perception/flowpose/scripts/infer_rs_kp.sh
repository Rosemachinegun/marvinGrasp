python py_runners/infer_rs_kp.py \
  --tracking \
  --realsense \
  --pretrained_flow_model_path results/ckpts/test/ckpt_epoch283.pth \
  --pretrained_scale_model_path results/ckpts/test/ckpt_epoch50_mixed.pth \
  --pretrained_kp_model_path results/ckpts/Keypoint/best_model.pth \
  --device cuda \
  --frame_gap_threshold 10 \
  --show
