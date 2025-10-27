from torchvision.datasets import FashionMNIST # 从 torchvision.datasets 模块中导入 FashionMNIST 数据集
from torchvision import transforms # 导入 torchvision.transforms 模块，用于图像变换
import numpy as np # 导入 numpy 模块，用于数值计算
import torch.utils.data as data # 导入 torch.utils.data 模块，用于处理数据集
import matplotlib.pyplot as plt # 导入 matplotlib.pyplot 模块，用于可视化

from model import LeNet # 从 model.py 中导入 LeNet 模型
import torch # 导入 torch 模块，用于张量计算
from torch import nn # 导入 torch.nn 模块，用于定义神经网络层
import copy # 导入 copy 模块，用于复制对象
import pandas as pd

# 定义一个函数，用于处理测试集的数据
def deal_test_data():
    # 加载FashionMNIST数据集, root 是数据集的根目录，train=False 表示加载测试集，download=True 表示如果数据集不存在，就从网上下载,
    # transform 是对数据进行预处理的操作，这里是将图像 resize 到 28x28 大小，然后转换为张量，为了和模型的输入大小一致
    # 这里的 transforms.Compose 是将多个预处理操作组合起来，这里是将 Resize 和 ToTensor 操作组合起来，先将图像 resize 到 28x28 大小，然后转换为张量
    dataset = FashionMNIST(root='./data',
                              train=False,
                              download=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]))


    # 定义数据加载器, DataLoader 是一个迭代器，它的作用是将数据集分成多个 batch，每个 batch 包含多个样本，这里是每个 batch 包含 64 个样本
    # shuffle=True 表示在每个 epoch 开始时，将数据集随机打乱，num_workers=4 表示使用 4 个线程来加载数据，加快加载速度
    # 定义测试集的数据加载器, 测试集的数据加载器的作用是将测试集分成多个 batch，每个 batch 包含 64 个样本，每个样本是一个 28x28 的图像和一个标签
    test_dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    return test_dataloader

# test_dataloader = deal_test_data()

# 定义一个函数，用于测试模型
def test_model(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将模型移动到 指定的设备上
    model = model.to(device)
    #
    test_dataloader = deal_test_data()
