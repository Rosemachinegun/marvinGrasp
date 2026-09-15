# FlowPose
```bash
conda create -n genpose2 python==3.10.14
conda activate flowpose
```

- ### Install pytorch
``` bash
conda install cudatoolkit=11
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit
```

``` bash
pip install -r requirements.txt 
pip install "setuptools<81"
pip install cutoop
pip install ultralytics
```

- ### Compile pointnet2

``` bash
cd networks/pts_encoder/pointnet2_utils/pointnet2
export CUDA_HOME=$CONDA_PREFIX
python setup.py install
```

- ### how to use
conda activate flowpose

## train flowpose with OMNI6D dataset
bash scripts/train_flow_raw.sh

## train flowpose with OMNI6D Webdataset (Compressed for optimized training speed)
bash scripts/train_flow_webdataset.sh

## using the flowpose on dataset (OMNI6D dataset format)
bash scripts/infer.sh
## using the flowpose on realsense
bash scripts/infer_rs.sh
## using the flowpose-kpd on realsense
bash scripts/infer_rs_kp.sh

## using the kp on dataset
bash scripts/infer_kp.sh

## evaluation on dataset (OMNI6D Webdataset dataset format)
bash scripts/eval.sh

