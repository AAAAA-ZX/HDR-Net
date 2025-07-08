# HDR-Net


## Citation


## How to Train/Finetune AMHDM-SRGAN

- [Train AMHDM-SRGAN](#train-amhdm-srgan)
  - [Overview](#overview)
  - [Dataset Preparation](#dataset-preparation)
  - [Train AMHDM-SRNet](#Train-AMHDM-SRNet)
  - [Train AMHDM-SRGAN](#Train-AMHDM-SRGAN)


### Environmental dependencies
- Python 3.8+ / PyTorch 1.9+ / CUDA 11.1
 Install commands:'pip install -r requirements.txt'

### Train HDR-Net
python /hy-tmp/HDnet/train.py


### Run HDR-Net
python /hy-tmp/HDnet/inference.py \--input /hy-tmp/HDnet/data/DIV2K_train_LR \--output /hy-tmp/HDnet/results \--model_dir /hy-tmp/HDnet/checkpoints
