DATA_PATH="/media/kewei/Training/dataROPE_Full"
CUDA_VISIBLE_DEVICES=0 python py_runners/train.py \
--arch pointnet \
--log_dir FlowNet \
--raw \
--data_path $DATA_PATH \
--batch_size 128 \
--n_epochs 20 \
--seed 0 \
--is_train \
--dino pointwise \
--num_workers 16 \
--lr 1e-3 \
