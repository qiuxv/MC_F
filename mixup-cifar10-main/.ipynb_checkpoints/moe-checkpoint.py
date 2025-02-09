import os
import numpy as np
import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns

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
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)

# 初始化 A 矩阵
A = np.zeros((10, 10), dtype=int)
B = np.zeros((10, 10), dtype=int)
Q = np.zeros((10000), dtype=int)
# 获取每个类别的样本索引
class_indices = {cls: [] for cls in range(10)}
for idx, (_, target) in enumerate(testset):
    class_indices[target].append(idx)

# 检查是否有缺失类别的样本
for cls, indices in class_indices.items():
    if len(indices) == 0:
        print(f"警告: 类别 {cls} 无样本数据。")


# mixup 数据生成
def mixup_data(img_a, img_b, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    mixed_img = lam * img_a + (1 - lam) * img_b
    return mixed_img, lam


# 生成 A 矩阵
rng = np.random.default_rng()  # 随机数生成器
i = 1   # 遍历类 i
for j in range(10):  # 遍历类 j
    for k in np.arange(0.20, 0.800, 0.01):  # k 从 0.4 到 0.6
        judge_i = 0
        judge_j = 0
        for w in range(1000):
            if len(class_indices[i]) == 0 or len(class_indices[j]) == 0:
                continue  # 如果某类无样本则跳过

            # 从每个类别随机抽取一张图片
            a_idx = rng.choice(class_indices[i])
            b_idx = rng.choice(class_indices[j])

            a_img, _ = testset[a_idx]
            b_img, _ = testset[b_idx]

            # 生成 mixup 图片
            c_img, lam = mixup_data(a_img, b_img, k)
            c_img = c_img.unsqueeze(0)  # 添加 batch 维度
            if use_cuda:
                c_img = c_img.cuda()

            # 模型预测
            with torch.no_grad():
                outputs = net(c_img)
                _, predicted = outputs.max(1)
            # 更新 A 矩阵
            if predicted.item() == i:
                judge_i += 1
            if predicted.item() == j:
                judge_j += 1
        B[1][j] = judge_i
        B[j][i] = judge_j
        if judge_i > judge_j:
            Q[(int)(k*100)] += judge_i
            A[i][j] += 1
        else:
            Q[(int)(k*100)] = 0
            A[j][i] += i
        print(i)
        print(" ")
        print(j)
        print(" ")
        print(judge_i)
        print(" ")
        print(judge_j)
        print(" ")
        print(k)
        print("one time\n")

# 创建结果目录
os.makedirs("results", exist_ok=True)

# 画热力图并保存
plt.figure(figsize=(10, 8))
sns.heatmap(A, annot=True, fmt="d", cmap="YlGnBu", cbar=True)
plt.title("Class MixUp Heatmap")
plt.xlabel("Class")
plt.ylabel("Class")
heatmap_path = "results/delete_1_v.png"
plt.savefig(heatmap_path)
plt.close()
print(f"热力图已保存到 {heatmap_path}")
# 画热力图并保存
plt.figure(figsize=(10, 8))
sns.heatmap(B, annot=True, fmt="d", cmap="YlGnBu", cbar=True)
plt.title("Class MixUp Heatmap")
plt.xlabel("Class")
plt.ylabel("Class")
heatmap_path = "results/delete_1_r.png"
plt.savefig(heatmap_path)
plt.close()
print(f"热力图已保存到 {heatmap_path}")
# -57 319 -213 -385 -237 -6 539 203 -105 -58     0.4-0.6
# -40 263 -60 -233 -170 73 132 49 36 -5          0.45-0.55
