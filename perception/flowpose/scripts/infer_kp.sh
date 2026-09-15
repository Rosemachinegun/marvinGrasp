python py_runners/kp_val.py \
  --tracking \
  --data_path /media/kewei/Training/dataKPT/set_03 \
  --pretrained_flow_model_path results/ckpts/FlowNet/scorenet_local.pth \
  --pretrained_scale_model_path results/ckpts/ScaleNet/scalenet_local.pth \
  --pretrained_kp_model_path results/ckpts/Keypoint/best_model.pth \
  --device cuda \
  --frame_gap_threshold 10 \
  --show
