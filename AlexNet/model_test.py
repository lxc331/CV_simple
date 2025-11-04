from pandas.core.dtypes.common import classes
from torchvision.datasets import FashionMNIST # 从 torchvision.datasets 模块中导入 FashionMNIST 数据集
from torchvision import transforms # 导入 torchvision.transforms 模块，用于图像变换
import numpy as np # 导入 numpy 模块，用于数值计算
import torch.utils.data as data # 导入 torch.utils.data 模块，用于处理数据集
import matplotlib.pyplot as plt # 导入 matplotlib.pyplot 模块，用于可视化

from model import AlexNet # 从 model.py 中导入 AlexNet 模型
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
                              transform=transforms.Compose([transforms.Resize(size=227), transforms.ToTensor()]))


    # 定义数据加载器, DataLoader 是一个迭代器，它的作用是将数据集分成多个 batch，每个 batch 包含多个样本，这里是每个 batch 包含 1 个样本
    # shuffle=True 表示在每个 epoch 开始时，将数据集随机打乱，num_workers=0 表示使用 0 个线程来加载数据，加快加载速度
    # 定义测试集的数据加载器, 测试集的数据加载器的作用是将测试集分成多个 batch，每个 batch 包含 64 个样本，每个样本是一个 28x28 的图像和一个标签
    test_dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    return test_dataloader

# 定义一个函数，用于测试模型
def test_model(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将模型移动到 指定的设备上
    model = model.to(device)
    # 初始化测试集的准确率为 0
    test_accuracy = 0.0
    # 初始化测试集的样本数为 0
    test_num = 0
    # 只进行前向传播，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 将测试集的样本移动到 指定的设备上
            test_data_x = test_data_x.to(device)
            # 将测试集的标签移动到 指定的设备上
            test_data_y = test_data_y.to(device)
            # 切换模型到评估模式, 评估模式下，模型的行为会和训练模式下不同, 例如 Dropout 层会被关闭, BatchNorm 层会使用训练时的统计信息
            # 这是因为在训练模式下，Dropout 层会随机将输入的一些元素设为 0，而在评估模式下，Dropout 层会将所有元素都设为 1，
            # 这是为了保持模型的稳定性，避免在评估时因为随机失活而导致的结果不一致
            model.eval()
            # 前向传播，计算模型的输出
            test_output = model(test_data_x)
            # 计算模型的预测结果，这里是取输出中概率最大的那个类作为预测结果
            # dim=1 表示在每一行（10个类别）中查找概率最大的索引, 作为模型的预测标签
            # argmax 函数返回的是概率最大的索引，这里是将索引转换为类别标签
            test_predict = torch.argmax(test_output, dim=1)
            # 计算测试集的准确率，这里是判断预测结果是否等于真实标签，如果相等，就将准确率加 1
            # 这里的 sum() 函数是将所有相等的元素的和相加，这里是将所有预测正确的元素的和相加，得到预测正确的样本数
            # 这里的 test_predict == test_data_y 是一个布尔张量，这里是将所有预测正确的元素设为 True，将所有预测错误的元素设为 False
            test_accuracy += torch.sum(test_predict == test_data_y)
            # 测试集的样本数加 1
            test_num += test_data_x.size(0)

    # 计算测试集的准确率，这里是将准确率除以样本数，得到准确率
    # 这里的 item() 函数是将张量转换为 Python 中的标量，这里是将预测正确的样本数转换为 Python 中的标量
    test_accuracy = test_accuracy.double().item() / test_num
    print(f"测试集的准确率为: {test_accuracy:.4f}")

# 定义一个函数，用于测试每个 batch(这里是 1 个样本) 上的预测与真实标签是否相等
def test_model_on_batch(model, test_dataloader):
    # 将模型移动到 指定的设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义类别名称, 这里是 FashionMNIST 数据集的类别名称
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # 只进行前向传播，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad(): # no_grad 上下文管理器, 用于在不计算梯度的情况下进行前向传播, 从而节省内存, 加快运行速度
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 开启评估模式, 评估模式下，模型的行为会和训练模式下不同, 例如 Dropout 层会被关闭, BatchNorm 层会使用训练时的统计信息
            model.eval()
            # 前向传播，计算模型的输出
            output = model(b_x)
            # 计算模型的预测结果，这里是取输出中概率最大的那个类作为预测结果
            # dim=1 表示在每一行（10个类别）中查找概率最大的索引, 作为模型的预测标签
            # argmax 函数返回的是概率最大的索引，这里是将索引转换为类别标签
            pre_label = torch.argmax(output, dim=1)
            # 也可以将模型的预测结果转换为 numpy 数组(result_label = pre_label.numpy()), 将模型的预测结果从 torch 张量转换为 numpy 数组
            # 结果是还原成了numpy数组[]
            # 这里是将模型的预测结果从 torch 张量转换为 Python 中的标量(单个元素), 方便后续的分析和可视化
            result_label = pre_label.item()
            # 这里是将模型的真实标签从 torch 张量转换为 Python 中的标量(单个元素个, 方便后续的分析和可视化
            label = b_y.item()
            print(f'预测值: {classes[result_label]},-------,  真实值: {classes[label]}')


if __name__ == '__main__':
    # 加载测试集的数据加载器
    test_dataloader = deal_test_data()
    # 加载模型
    model = AlexNet()
    # 加载模型的参数
    model.load_state_dict(torch.load('./model/best_model.pth'))
    # 测试模型在每个 batch 上的准确率
    test_model_on_batch(model, test_dataloader)
    # 测试模型总的准确率
    test_model(model, test_dataloader)

