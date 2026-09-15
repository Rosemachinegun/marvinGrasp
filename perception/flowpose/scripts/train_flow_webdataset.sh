SHARD_PATH="/media/kewei/Training/dataSOPE_webdataset/train-{000000..000937}.tar"
CUDA_VISIBLE_DEVICES=0 python py_runners/train.py \
--arch pointnet \
--log_dir FlowNet \
--shard_path $SHARD_PATH \
--batch_size 128 \
--n_epochs 20 \
--seed 0 \
--is_train \
--dino pointwise \
--num_workers 16 \
--lr 1e-3 \