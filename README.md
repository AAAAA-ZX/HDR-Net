# HDR-Net


## Citation
If you have used this code or data in your research, please cite the following papers:
```bibtex
@article{
  title    = {Enhanced Blind Super-Resolution via Hybrid Degradation Estimation and Dynamic Residual Networks},  
  author   = {Zhang, Xin and Yi, Huawei and Zhao, Mengyuan and Wang, Yanfei and Zhang, Linchen},
  journal  = {The Visual Computer},
  year     = {2025},
}
Zhang, X., Yi, H., Zhao, M. et al. Enhanced Blind Super-Resolution via Hybrid Degradation Estimation and Dynamic Residual Networks. Vis Comput(2025).
```



### Environmental dependencies
- Python 3.8+ / PyTorch 1.9+ / CUDA 11.1
 Install commands:'pip install -r requirements.txt'

#### Dataset Preparation

We use DIV2K  datasets for our training. Only HR images are required. <br>
You can download from :
Train Dataset
1. DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/
2. Flickr2K: https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
Test Dataset
Set5: https://github.com/jbhuang0604/SelfExSR
Set14: https://github.com/jbhuang0604/SelfExSR
BSD100: https://github.com/jbhuang0604/SelfExSR
Urban100: https://github.com/jbhuang0604/SelfExSR
RealSR:https://drive.google.com/file/d/1gKnm9BdgyqISCTDAbGbpVitT-QII_unw/view?usp=drive_open
DRealSR:https://drive.google.com/drive/folders/1tP5m4k1_shFT6Dcw31XV8cWHtblGmbOk
Here are steps for data preparation.

### Train HDR-Net
python /hy-tmp/HDnet/train.py


### Run HDR-Net
python /hy-tmp/HDnet/inference.py \--input /hy-tmp/HDnet/data/DIV2K_train_LR \--output /hy-tmp/HDnet/results \--model_dir /hy-tmp/HDnet/checkpoints
