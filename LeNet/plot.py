from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import torch.utils.data as data

# 加载FashionMNIST数据集, root 是数据集的根目录，train=True 表示加载训练集，download=True 表示如果数据集不存在，就从网上下载,
# transform 是对数据进行预处理的操作，这里是将图像 resize 到 224x224 大小，然后转换为张量
# 这里的 transforms.Compose 是将多个预处理操作组合起来，这里是将 Resize 和 ToTensor 操作组合起来，先将图像 resize 到 224x224 大小，然后转换为张量
train_data = FashionMNIST(root='./data',
                          train=True,
                          download=True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]))

# 定义训练集的 DataLoader,加载训练集, batch_size 是每个批次的样本数量，shuffle=True 表示在每个 epoch 开始前，打乱样本的顺序，num_workers 是加载数据的线程数量
train_loader = data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)

for step, (b_x,b_y) in enumerate(train_loader):
    if step > 0:
        break
    batch_x = b_x.squeeze().numpy() # 从四维张量中移除第一维，然后转换为 numpy 数组
    batch_y = b_y.numpy() # 将标签张量转换为 numpy 数组
    class_label = train_data
