from torch.utils.data import Dataset
from PIL import Image
import glob
import os
from torchvision import transforms 
import torchvision.transforms.functional as TF
import random

class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, upscale=4, transform=None):
        """
        Args:
            hr_dir: 高分辨率图像目录
            lr_dir: 低分辨率图像目录
            upscale: 超分辨率倍数（默认4）
            transform: 数据增强操作
        """
        self.hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
        self.lr_files = sorted(glob.glob(os.path.join(lr_dir, "*.png")))
        self.upscale = upscale  # 显式定义超分倍数
        self.transform = transform

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr = Image.open(self.hr_files[idx]).convert("RGB")
        lr = Image.open(self.lr_files[idx]).convert("RGB")
        
        # 强制验证尺寸关系
        assert hr.size == (lr.size[0] * self.upscale, lr.size[1] * self.upscale), \
            f"尺寸不匹配: HR {hr.size} vs LR {lr.size} (需满足{self.upscale}倍关系)"
        
        # 同步随机裁剪（HR尺寸为512，LR自动为128）
        if self.transform:
            # HR裁剪参数
            i, j, h, w = transforms.RandomCrop.get_params(hr, output_size=(512, 512))
            hr = TF.crop(hr, i, j, h, w)
            
            # 对应LR裁剪参数
            lr_i, lr_j = i // self.upscale, j // self.upscale
            lr_h, lr_w = 512 // self.upscale, 512 // self.upscale
            lr = TF.crop(lr, lr_i, lr_j, lr_h, lr_w)
        
        # 转换为Tensor
        hr = TF.to_tensor(hr)
        lr = TF.to_tensor(lr)
        return {"lr": lr, "hr": hr}

