import argparse
import torch
from PIL import Image
from torchvision import transforms
from models.generator import Generator
from models.degradation_estimator import DegradationEstimator

# ==============================
# 参数解析
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="输入低分辨率图像路径")
parser.add_argument("--output", type=str, default="output.png", help="/hy-tmp/HDnet/data/results")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="模型权重目录")
opt = parser.parse_args()


# ==============================
# 图像预处理
# ==============================
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为[0,1]范围的张量
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
    ])
    return transform(image).unsqueeze(0)  # 增加batch维度


# ==============================
# 模型加载
# ==============================
def load_models(model_dir):
    # 加载退化估计网络
    degradation_estimator = DegradationEstimator()
    degradation_estimator.load_state_dict(torch.load(f"{model_dir}/degradation_estimator.pth"))

    # 加载超分辨率生成器
    generator = Generator()
    generator.load_state_dict(torch.load(f"{model_dir}/generator.pth"))

    return degradation_estimator.eval(), generator.eval()


# ==============================
# 主推理流程
# ==============================
def main():
    # 加载模型
    degradation_estimator, generator = load_models(opt.model_dir)
    degradation_estimator.cuda()
    generator.cuda()

    # 预处理输入图像
    lr_image = preprocess_image(opt.input).cuda()

    # 退化估计
    with torch.no_grad():
        d = degradation_estimator(lr_image)

    # 超分辨率重建
    with torch.no_grad():
        sr_image = generator(lr_image, d)

    # 后处理
    sr_image = sr_image.squeeze(0).clamp(-1, 1)  # 去除batch维度并截断
    sr_image = (sr_image * 0.5 + 0.5) * 255.0  # 反标准化并转换为[0,255]
    sr_image = transforms.ToPILImage()(sr_image.byte().cpu())

    # 保存结果
    sr_image.save(opt.output)
    print(f"重建完成！结果保存至: {opt.output}")


if __name__ == "__main__":
    main()