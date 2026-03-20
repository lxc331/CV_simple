from torchvision.datasets import ImageFolder  # 从 torchvision.datasets 模块中导入 FashionMNIST 数据集
from torchvision import transforms # 导入 torchvision.transforms 模块，用于图像变换
import numpy as np # 导入 numpy 模块，用于数值计算
import torch.utils.data as data # 导入 torch.utils.data 模块，用于处理数据集
import matplotlib.pyplot as plt # 导入 matplotlib.pyplot 模块，用于可视化

from model import GoogLeNet, Inception # 从 model.py 中导入 GoogLeNet和 Inception 模型
import torch # 导入 torch 模块，用于张量计算
from torch import nn # 导入 torch.nn 模块，用于定义神经网络层
import copy # 导入 copy 模块，用于复制对象
import pandas as pd
from PIL import Image # 导入 PIL.Image 模块，用于加载图像

# 单个图像数据加载
def deal_a_image_data():
    # 做单张图像的预测推理
    image = Image.open('./data/806.jpg')
    # 对图像变换为 224x224 大小, 将图像转换为张量, 并归一化到 [0, 1] 范围（利用正态分布归一化）
    # 使用的均值和标准差是通过 mean_std.py 计算得到的
    normalize = transforms.Normalize(mean=[0.4853, 0.4523, 0.4146], std=[0.2617, 0.2544, 0.2580])
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    # 对图像进行变换（因为这里是单个图像没有批次的维度，所以需要添加一个维度, 即添加一个批次）
    image = test_transform(image)
    # 将图像转换为批次格式, 即添加一个维度, 用于后续的前向传播(因为模型的输入一般是带有批次格式的张量)
    image = image.unsqueeze(0)

    return image, test_transform

def model_inference_a_image(model, image, test_transform):
    # 将模型移动到 指定的设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义类别名称, 这里是数据集的类别名称
    classes = ['Cat', 'Dog']
    # 前向传播，计算模型的输出(开始推理)
    with torch.no_grad():
        # 将模型设置为评估模式
        model.eval()
        # 将图像移动到 指定的设备上
        image = image.to(device)
        # 前向传播，计算模型的输出
        output = model(image)
        # 获取模型的预测标签
        pre_label = torch.argmax(output, dim=1)
        # 打印模型的预测标签
        result = classes[pre_label.item()]
        print(result)

if __name__ == '__main__':
    # 加载模型
    model = GoogLeNet(Inception)
    # 加载模型的参数
    model.load_state_dict(torch.load('./model/best_model.pth'))
    # 加载测试集的图像数据
    image, test_transform = deal_a_image_data()
    # 对图像进行推理
    model_inference_a_image(model, image, test_transform)
