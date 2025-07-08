import torch
import torch.nn as nn
import torch.fft as fft
from torchvision import models
from torch.nn.functional import interpolate

class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, is_real):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.loss(pred, target)

class MultiScalePerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用VGG19权重（修正预训练参数加载）
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False

        # 定义多尺度特征层（对应论文中的conv1_2, conv2_2, conv3_4, conv4_4）
        self.layer_indices = [3, 8, 17, 26, 35]  # 修正层索引
        self.vgg_slices = nn.ModuleList([
            nn.Sequential(*[vgg[i] for i in range(idx+1)])
            for idx in self.layer_indices
        ])
        self.weights = [0.1, 0.2, 0.3, 0.2, 0.2]  # 论文指定权重

    def forward(self, sr, hr):
        # 处理单通道图像（复制到3通道）
        if sr.shape[1] == 1:
            sr = sr.repeat(1, 3, 1, 1)
            hr = hr.repeat(1, 3, 1, 1)
        
        loss = 0
        for slice, weight in zip(self.vgg_slices, self.weights):
            sr_features = slice(sr)
            hr_features = slice(hr)
            loss += weight * torch.mean(torch.abs(sr_features - hr_features))
        return loss

class FrequencyDomainLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, sr, hr):
        # 动态调整FFT分辨率（论文新增）
        sr = interpolate(sr, scale_factor=0.5, mode='bicubic')  # 降低计算量
        hr = interpolate(hr, scale_factor=0.5, mode='bicubic')
        
        # 计算傅里叶幅度谱
        sr_fft = fft.fftshift(fft.fft2(sr, norm='ortho'))
        hr_fft = fft.fftshift(fft.fft2(hr, norm='ortho'))
        
        # 高频区域约束（论文创新点）
        mask = torch.ones_like(sr_fft)
        mask[:, :, :mask.shape[2]//4, :mask.shape[3]//4] = 0  # 屏蔽低频区域
        return self.criterion(sr_fft*mask, hr_fft*mask)

class CombinedLoss(nn.Module):
    def __init__(self, loss_weights):
        super().__init__()
        self.loss_weights = loss_weights
        self.pixel_loss = nn.L1Loss()
        self.perceptual_loss = MultiScalePerceptualLoss()
        self.frequency_loss = FrequencyDomainLoss()
        self.gan_loss = GANLoss()

    def forward(self, sr, hr, real_logits=None, fake_logits=None):
        # 像素损失（论文权重0.1）
        loss_pixel = self.pixel_loss(sr, hr) * self.loss_weights.get('pixel', 0.1)
        
        # 多尺度感知损失（论文权重0.3）
        loss_perceptual = self.perceptual_loss(sr, hr) * self.loss_weights.get('perceptual', 0.3)
        
        # 频域约束损失（论文权重0.2）
        loss_frequency = self.frequency_loss(sr, hr) * self.loss_weights.get('frequency', 0.2)
        
        # 对抗损失（论文权重0.01）
        loss_gan = 0
        if fake_logits is not None:
            loss_gan = self.gan_loss(fake_logits, True) * self.loss_weights.get('gan', 0.01)
        
        return loss_pixel + loss_perceptual + loss_frequency + loss_gan

    # 独立判别器损失计算（论文公式）
    def discriminator_loss(self, real_logits, fake_logits):
        loss_real = self.gan_loss(real_logits, True)
        loss_fake = self.gan_loss(fake_logits, False)
        return loss_real + loss_fake