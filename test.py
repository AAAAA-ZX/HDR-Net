import os
import glob
import torch
from torchvision.utils import save_image
from models.generator import Generator
from models.degradation_estimator import DegradationEstimator
from utils.dataset import SRDataset
from utils.metrics import calculate_psnr, calculate_ssim
from torchvision import transforms

# 加载测试数据集
test_dataset = SRDataset(
    hr_dir="data/Set5/HR",
    lr_dir="data/Set5/LR",
    transform=transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

# 加载预训练模型
degradation_estimator = DegradationEstimator().cuda()
generator = Generator().cuda()
degradation_estimator.load_state_dict(torch.load("degradation_estimator.pth"))
generator.load_state_dict(torch.load("generator.pth"))

# 测试循环
for batch in test_loader:
    lr, hr = batch["lr"].cuda(), batch["hr"].cuda()
    d = degradation_estimator(lr)
    sr = generator(lr, d)

    # 计算指标
    psnr = calculate_psnr(sr, hr)
    ssim = calculate_ssim(sr, hr)
    print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

    # 保存结果
    save_image(sr, "output.png")