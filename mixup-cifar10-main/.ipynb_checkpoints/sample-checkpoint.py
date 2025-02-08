import os
import numpy as np
import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import save_image

# 确保使用CUDA
use_cuda = torch.cuda.is_available()

# 加载模型
checkpoint_path = './checkpoint/ckpt.t70_0'
assert os.path.exists(checkpoint_path), "Checkpoint 文件不存在，请检查路径。"
checkpoint = torch.load(checkpoint_path)
net = checkpoint['net']
net.eval()
if use_cuda:
    net.cuda()

# 数据预处理和加载
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = datasets.CIFAR10(root='~/data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)

# 获取每个类别的样本索引
class_indices = {cls: [] for cls in range(10)}
for idx, (_, target) in enumerate(testset):
    class_indices[target].append(idx)


# mixup 数据生成
def mixup_data(img_a, img_b, alpha):
    lam = alpha
    mixed_img = lam * img_a + (1 - lam) * img_b
    return mixed_img


rng = np.random.default_rng()
# 随机数生成器
i = 9
Q = 0
j = 3
k: float = 0.424
for w in range(5000):

    # 从每个类别随机抽取一张图片
    a_idx = rng.choice(class_indices[i])
    b_idx = rng.choice(class_indices[j])

    a_img, _ = testset[a_idx]
    b_img, _ = testset[b_idx]

    # 生成 mixup 图片
    c_img = mixup_data(a_img, b_img, k)
    c_img = c_img.unsqueeze(0)  # 添加 batch 维度
    if use_cuda:
        c_img = c_img.cuda()

    # 模型预测
    with torch.no_grad():
        outputs = net(c_img)
        _, predicted = outputs.max(1)

    if predicted.item() == i:
        save_image(c_img, "results/sam2/"+str(Q) + 'image.png')  # 替换为您想要保存图像的位置
        Q += 1
print(Q)


