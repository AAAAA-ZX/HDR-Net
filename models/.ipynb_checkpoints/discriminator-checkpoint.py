import torch
import torch.nn as nn
from models.blocks import DynamicConv2d, MSRB

class Discriminator(nn.Module):
    def __init__(self, nf=64, degradation_dim=64):
        super().__init__()
        self.degradation_dim = degradation_dim
        
        # 动态卷积适配模块（轻量化设计）
        self.dynamic_adapter = nn.Sequential(
            nn.Linear(degradation_dim, nf),
            nn.ReLU(),
            nn.Linear(nf, nf)
        )
        
        # VGG风格的卷积块（逐步下采样）
        self.features = nn.Sequential(
            # Block 1: 动态卷积层
            DynamicConv2d(3, nf, 3, 1, 1),  # 根据退化特征调整卷积核
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, 3, 2, 1),      # 下采样2倍
            
            # Block 2: 多尺度残差块
            MSRB(nf),  # 提取模糊/噪声等退化相关特征
            nn.Conv2d(nf, nf*2, 3, 2, 1),   # 下采样4倍
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 3: 深层特征提取
            nn.Conv2d(nf*2, nf*4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*4, 3, 2, 1), # 下采样8倍
            
            # Block 4: 全局特征压缩
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nf*4, nf*8, 1),       # 通道压缩
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 分类头（轻量化设计）
        self.classifier = nn.Sequential(
            nn.Linear(nf*8, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1)
        )

    def forward(self, x, degradation_features):
        # 退化特征适配
        d = self.dynamic_adapter(degradation_features)
        
        # 动态卷积前向传播
        x = self.features[0](x, d)  # 动态卷积层
        x = self.features[1](x)     # 激活
        
        # 剩余VGG块前向传播
        for block in self.features[2:]:
            x = block(x)
        
        # 全局池化与分类
        x = x.view(x.size(0), -1)
        return self.classifier(x)