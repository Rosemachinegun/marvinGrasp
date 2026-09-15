python py_runners/infer.py \
  --data_path /media/kewei/Training/dataSOPE_00_eval/0103 \
  --tracking \
  --pretrained_flow_model_path results/ckpts/FlowNet/scorenet_local.pth \
  --pretrained_scale_model_path results/ckpts/ScaleNet/scalenet_local.pth \
  --device cuda \
  --frame_gap_threshold 10 \
  --show \
  --data_mode isaac \
