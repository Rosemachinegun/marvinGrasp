SHARD_PATH="/media/kewei/Training/dataSOPE_webdataset/train-{000000..000937}.tar"
CUDA_VISIBLE_DEVICES=0 python py_runners/train.py \
--arch scalenet \
--log_dir ScaleNet \
--agent_type scale \
--shard_path $SHARD_PATH \
--batch_size 128 \
--n_epochs 5 \
--seed 0 \
--is_train \
--dino pointwise \
--num_workers 12 \
--lr 1e-3 \
--pretrained_flow_model_path results/pretrained_ckpts/FlowNet/scorenet_local.pth \
--pretrained_scale_model_path results/pretrained_ckpts/ScaleNet/scalenet.pth \