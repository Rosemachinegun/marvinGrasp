SHARD_PATH="/media/kewei/Training/dataSOPE_webdataset/train-{000000..000002}.tar"
python py_runners/eval_on_webdataset.py \
  --pretrained_flow_model_path results/ckpts/FlowNet/scorenet_local.pth \
  --pretrained_scale_model_path results/ckpts/ScaleNet/scalenet_local.pth \
  --device cuda \
  --shard_path $SHARD_PATH \
  --seed 0 \
  --frame_gap_threshold 10 \
  --batch_size 32 \
  --dino pointwise \
  --num_worker 2 \
  --drop_step 50 \
  --retain_ratio 1 \