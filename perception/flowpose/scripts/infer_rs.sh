python py_runners/infer_rs.py \
  --tracking \
  --realsense \
  --pretrained_flow_model_path results/ckpts/0605/ckpt_epoch91.pth \
  --pretrained_scale_model_path results/ckpts/0605/ckpt_epoch29.pth\
  --device cuda \
  --frame_gap_threshold 10 \
  --show \
  --data_mode rs \

  # --pretrained_flow_model_path results/ckpts/FlowNet/ckpt_epoch180.pth \
  # --pretrained_scale_model_path results/ckpts/ScaleNet/ckpt_epochqyy.pth \

  # --pretrained_flow_model_path results/ckpts/FlowNet/scorenet_local.pth \
  # --pretrained_scale_model_path results/ckpts/ScaleNet/scalenet_local.pth \

  # --pretrained_flow_model_path results/ckpts/test/ckpt_epoch283.pth \
  # --pretrained_scale_model_path results/ckpts/test/ckpt_epoch50_mixed.pth\
