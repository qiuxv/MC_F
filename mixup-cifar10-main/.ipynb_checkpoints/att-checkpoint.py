#!/usr/bin/env python3
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt

# 直接从训练代码导入模型定义
from models.resnet import ResNet18

def load_model():
    # 创建模型实例
    model = ResNet18()
    
    # 加载检查点
    checkpoint = torch.load("./checkpoint/ckpt.t70_0")
    
    # 处理DataParallel前缀
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['net'].state_dict().items()}
    
    # 加载参数
    model.load_state_dict(state_dict)
    return model.cuda().eval()

# 定义标准化参数（CPU版本）
mean_cpu = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
std_cpu = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

# 定义GPU版本标准化参数
mean_cuda = mean_cpu.cuda()
std_cuda = std_cpu.cuda()

def denormalize(tensor):
    """将张量移到CPU并反标准化"""
    return tensor.cpu() * std_cpu + mean_cpu

def pgd_attack(model, image, target, epsilon=8/255, alpha=2/255, iterations=20):
    image = image.clone().detach().cuda()  # 确保输入在GPU
    delta = torch.zeros_like(image, requires_grad=True).cuda()
    
    for _ in range(iterations):
        adv_image = image + delta
        adv_image = torch.clamp(adv_image, 0, 1)
        
        # 使用GPU进行标准化
        norm_adv = (adv_image - mean_cuda) / std_cuda
        
        outputs = model(norm_adv)
        loss = nn.CrossEntropyLoss()(outputs, target)
        
        loss.backward()
        delta.data = (delta + alpha * delta.grad.sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    
    final_adv = torch.clamp(image + delta, 0, 1)
    return (final_adv - mean_cuda) / std_cuda

def generate_adv_samples():
    save_dir = "adv_samples"
    os.makedirs(save_dir, exist_ok=True)
    
    model = load_model()
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    cat_samples = [x for x in testset if x[1] == 3][:50]  # 类别3=猫
    
    success = 0
    for idx, (img, _) in enumerate(cat_samples):
        # 数据加载到GPU
        image = img.unsqueeze(0).cuda()
        target = torch.tensor([9], device="cuda")  # 目标类别9=卡车
        
        # 生成对抗样本（保持在GPU）
        adv_img = pgd_attack(model, image, target)
        
        # 验证结果
        with torch.no_grad():
            pred = model(adv_img).argmax()
            success += int(pred == 9)
        
        # 反标准化并保存到CPU
        orig = denormalize(image.squeeze())
        adv = denormalize(adv_img.squeeze())
        
        save_image(orig, f"{save_dir}/orig_{idx}.png")
        save_image(adv, f"{save_dir}/adv_{idx}.png")
        
        # 可视化对比
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1).imshow(orig.permute(1,2,0))
        plt.title("Original Cat").axis("off")
        plt.subplot(1,2,2).imshow(adv.permute(1,2,0))
        plt.title("Adversarial Truck").axis("off")
        plt.savefig(f"{save_dir}/compare_{idx}.png", bbox_inches="tight")
        plt.close()

    print(f"攻击成功率: {success}/{len(cat_samples)} ({success/len(cat_samples)*100:.1f}%)")

if __name__ == "__main__":
    generate_adv_samples()