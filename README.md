# HDR-Net


## Citation




### Environmental dependencies
- Python 3.8+ / PyTorch 1.9+ / CUDA 11.1
 Install commands:'pip install -r requirements.txt'

#### Dataset Preparation

We use DIV2K  datasets for our training. Only HR images are required. <br>
You can download from :

1. DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/
2. Flickr2K: https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar

Here are steps for data preparation.

### Train HDR-Net
python /hy-tmp/HDnet/train.py


### Run HDR-Net
python /hy-tmp/HDnet/inference.py \--input /hy-tmp/HDnet/data/DIV2K_train_LR \--output /hy-tmp/HDnet/results \--model_dir /hy-tmp/HDnet/checkpoints
