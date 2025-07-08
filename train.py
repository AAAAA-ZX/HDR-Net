# train.py
import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.generator import Generator
from models.discriminator import Discriminator
from models.degradation_estimator import DegradationEstimator
from utils.loss import CombinedLoss
from torchvision import transforms
from utils.dataset import SRDataset
from torch.cuda.amp import GradScaler, autocast

# ==============================
# 读取配置文件
# ==============================
with open("/hy-tmp/HDnet/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 创建模型保存目录
checkpoint_dir = config["training"].get("checkpoint_dir", "./checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"模型将保存至: {checkpoint_dir}")

# ==============================
# 数据加载
# ==============================
train_dataset = SRDataset(
    hr_dir=config["data"]["train_hr_dir"],
    lr_dir=config["data"]["train_lr_dir"],
    upscale=config["data"]["upscale_factor"],
    transform=transforms.Compose([
        transforms.RandomCrop(512),
        transforms.ToTensor()
    ])
)
train_loader = DataLoader(
    train_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    drop_last=True,
    num_workers=4,
    pin_memory=True
)

# ==============================
# 模型初始化（显存优化）
# ==============================
degradation_estimator = DegradationEstimator().cuda()
generator = Generator(
    nf=config["network"]["nf"],  # 32
    nb=config["network"]["num_blocks"],  # 16
    degradation_dim=config["network"]["degradation_dim"]  # 128
).cuda()
discriminator = Discriminator(nf=16).cuda()

# ==============================
# 损失与优化器
# ==============================
criterion = CombinedLoss(config["loss_weights"]).cuda()
scaler = GradScaler()

optimizer_G = optim.Adam(
    list(generator.parameters()) + list(degradation_estimator.parameters()),
    lr=config["training"]["learning_rate"],
    betas=(0.9, 0.99)
)
optimizer_D = optim.Adam(
    discriminator.parameters(),
    lr=config["training"]["learning_rate"],
    betas=(0.9, 0.99)
)

# 调度器初始化（在训练循环外）
scheduler_G = optim.lr_scheduler.StepLR(
    optimizer_G,
    step_size=config["training"]["decay_steps"],  # 200轮
    gamma=config["training"]["decay_rate"]  # 0.5
)
scheduler_D = optim.lr_scheduler.StepLR(
    optimizer_D,
    step_size=config["training"]["decay_steps"],
    gamma=config["training"]["decay_rate"]
)

# ==============================
# 训练循环（严格遵循优化器→调度器顺序）
# ==============================
for epoch in range(config["training"]["num_epochs"]):
    for batch in train_loader:
        lr, hr = batch["lr"].cuda(non_blocking=True), batch["hr"].cuda(non_blocking=True)
        
        # 生成器前向传播（混合精度）
        with autocast():
            degradation_params = degradation_estimator(lr)
            sr = generator(lr, degradation_params)
        
        # 判别器前向传播
        real_logits = discriminator(hr)
        fake_logits = discriminator(sr.detach())
        
        # 生成器损失与优化（先优化）
        with autocast():
            loss_G = criterion(sr, hr, real_logits, fake_logits, degradation_params)
         # 生成器优化（先优化器更新）
        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)  # 优化器更新
        scaler.update()
        optimizer_G.zero_grad(set_to_none=True)
        
        # 判别器优化（先优化器更新）
        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D)  # 优化器更新
        scaler.update()
        optimizer_D.zero_grad(set_to_none=True)
    
    # 调度器更新（在优化器之后）
    scheduler_G.step()
    scheduler_D.step()
    
    # 模型保存
    if (epoch + 1) % config["training"]["save_interval"] == 0:
        torch.save(generator.state_dict(), f"{checkpoint_dir}/generator_epoch_{epoch+1}.pth")
        torch.save(degradation_estimator.state_dict(), f"{checkpoint_dir}/degradation_estimator_epoch_{epoch+1}.pth")
        torch.cuda.empty_cache()