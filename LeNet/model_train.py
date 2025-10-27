import time

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

# 定义一个函数，用于处理训练集和验证集的数据
def deal_train_and_val_data():
    # 加载FashionMNIST数据集, root 是数据集的根目录，train=True 表示加载训练集，download=True 表示如果数据集不存在，就从网上下载,
    # transform 是对数据进行预处理的操作，这里是将图像 resize 到 28x28 大小，然后转换为张量，为了和模型的输入大小一致
    # 这里的 transforms.Compose 是将多个预处理操作组合起来，这里是将 Resize 和 ToTensor 操作组合起来，先将图像 resize 到 28x28 大小，然后转换为张量
    dataset = FashionMNIST(root='./data',
                              train=True,
                              download=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]))

    # 随机划分训练集和验证集, random_split 函数的作用是将数据集随机划分成训练集和验证集，这里是将数据集随机划分成 80% 训练集和 20% 验证集
    train_data, val_data = data.random_split(dataset, [round(len(dataset) * 0.8), round(len(dataset) * 0.2)])

    # 定义数据加载器, DataLoader 是一个迭代器，它的作用是将数据集分成多个 batch，每个 batch 包含多个样本，这里是每个 batch 包含 64 个样本
    # shuffle=True 表示在每个 epoch 开始时，将数据集随机打乱，num_workers=4 表示使用 4 个线程来加载数据，加快加载速度
    # 定义训练集的数据加载器, 训练集的数据加载器的作用是将训练集分成多个 batch，每个 batch 包含 64 个样本，每个样本是一个 28x28 的图像和一个标签
    train_dataloader = data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    val_dataloader = data.DataLoader(val_data, batch_size=64, shuffle=True, num_workers=4)

    return train_dataloader, val_dataloader

# 定义一个函数，用于训练模型
def train_model(model, train_dataloader, val_dataloader, epochs):
    # 定义设备, 这里是使用 GPU 来训练模型, 如果 GPU 不可用, 就使用 CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 定义损失函数, 这里是使用交叉熵损失函数, 它的作用是计算模型输出和真实标签之间的差异, 用于训练模型
    criterion = nn.CrossEntropyLoss()
    # 定义优化器, 这里是使用 Adam 优化器, 它的作用是根据损失函数的梯度来更新模型的参数, 用于训练模型, 学习率为 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 将模型移动到 GPU 上或者 CPU 上
    model = model.to(device)
    # 定义最佳模型的权重, 这里是将当前模型的权重复制给 best_model_wts, 用于后续的模型保存
    best_model_wts = copy.deepcopy(model.state_dict())
    # 定义最佳准确率, 这里是将最佳准确率初始化为 0.0, 用于后续的模型保存
    best_acc = 0.0
    # 定义训练集和验证集的损失函数和准确率列表, 记录每轮训练集的损失值
    train_loss_all = []
    # 定义验证集的损失函数和准确率列表, 记录每轮验证集的损失值
    val_loss_all = []
    # 定义训练集和验证集的准确率列表, 记录每轮训练集的准确率
    train_acc_all = []
    # 定义验证集的准确率列表, 记录每轮验证集的准确率
    val_acc_all = []
    # 定义训练开始的时间, 记录每轮训练的时间
    start_time = time.time()
    # 定义训练总时间, 记录训练模型的总时间
    total_time = 0

    for i in range(epochs):
        print('Epoch {}/{}'.format(i + 1, epochs))
        print('-' * 10)
        # 初始化训练集和验证集的损失值和准确率和样本数量为0
        train_loss = 0
        train_acc = 0
        train_num = 0
        val_loss = 0
        val_acc = 0
        val_num = 0

        # 训练模型
        # 取出训练集的一个 batch, b_x 是一个 batch 中的图像张量, b_y 是一个 batch 中的标签张量
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 设置模型为训练模式, 这是因为在训练模式下, 模型会启用 dropout 等正则化技术, 而在验证模式下, 模型会关闭这些技术
            model.train()
            # 前向传播, 计算模型输出, output 是一个 batch 中的模型输出, 每个元素是一个类别的概率分布, 这里是一个 64x10 的张量, 每个元素是一个类别的概率分布
            # 64 是一个 batch 中的样本数量, 每个样本是一个 28x28 的图像, 这里是一个 64x1x28x28 的张量, 每个元素是一个像素值;10 是类别数量, 这里是 10 个类别（0-9）（最后一层有10个神经元）
            output = model(b_x)
            # 查找每一行（10个类别）中概率最大的索引, 作为模型的预测标签, pre_lab 是一个 batch 中的预测标签, 每个元素是一个类别的索引, 这里是一个 64 维的张量, 每个元素是一个类别的索引
            # dim=1 表示在每一行（10个类别）中查找概率最大的索引, 作为模型的预测标签
            pre_lab = torch.argmax(output, dim=1)
            # 计算损失函数（交叉熵损失函数）, loss 是一个 batch 中的平均损失值, 这里是一个标量, 表示模型输出和真实标签之间的差异
            loss = criterion(output, b_y)
            # 将梯度清零, 这是因为在每次反向传播之前, 都需要将梯度清零, 否则梯度会累加起来, 导致错误的更新
            optimizer.zero_grad()
            # 反向传播, 计算每个参数梯度
            loss.backward()
            # 利用 Adam 优化器更新模型参数, 这里是根据损失函数的梯度来更新模型的参数, 用于训练模型
            optimizer.step()

            # 对每个 batch 中的损失值和准确率进行累加, 用于计算训练集的损失值和准确率
            train_loss += loss.item() * b_x.size(0) # 每个 batch 中的平均损失值乘以 batch 中的样本数量, 用于计算训练集的损失值
            train_acc += torch.sum(pre_lab == b_y.data) # 如果预测标签和真实标签相等（预测正确）, 则累加 1, 否则累加 0
            train_num += b_x.size(0) # 每个 batch 中的样本数量, 用于计算训练集的样本数量

        # 验证模型
        # 取出训练集的一个 batch, b_x 是一个 batch 中的图像, b_y 是一个 batch 中的标签
        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 将模型设置为验证模式, 这是因为在验证模式下, 模型会关闭 dropout 等正则化技术, 而在训练模式下, 模型会启用这些技术
            model.eval()
            # 前向传播, 计算模型输出, output 是一个 batch 中的模型输出, 每个元素是一个类别的概率分布, 这里是一个 64x10 的张量, 每个元素是一个类别的概率分布
            output = model(b_x)
            # 查找每一行（10个类别）中概率最大的索引, 作为模型的预测标签, per_lab 是一个 batch 中的预测标签, 每个元素是一个类别的索引, 这里是一个 64 维的张量, 每个元素是一个类别的索引
            per_lab = torch.argmax(output, dim=1)

            # 注意：验证过程没有反向传播和参数更新

            # 计算损失函数（交叉熵损失函数）, val_loss 是一个 batch 中的损失值, 这里是一个标量, 表示模型输出和真实标签之间的差异
            val_loss += criterion(output, b_y).item() * b_x.size(0) # 每个 batch 中的损失值乘以 batch 中的样本数量, 用于计算验证集的损失值
            val_acc += torch.sum(per_lab == b_y.data) # 如果预测标签和真实标签相等（预测正确）, 则累加 1, 否则累加 0
            val_num += b_x.size(0) # 每个 batch 中的样本数量, 用于计算验证集的样本数量

        # 关系：一个数据集有多个 epoch, 每个 epoch 有多个 batch, 每个 batch 有多个（这里是64）样本
        # 计算每个 epoch 中的训练集和验证集的损失值和准确率, 并记录到列表中（注意：这里的损失值和准确率是每个 epoch 中的平均损失值和准确率）
        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        # 计算每个 epoch 中的训练集和验证集的准确率, 并记录到列表中（注意：这里的准确率是每个 epoch 中的准确率）
        train_acc_all.append(train_acc.double().item() / train_num)
        val_acc_all.append(val_acc.double().item() / val_num)
        # 打印每个 epoch 中的训练集和验证集的损失值和准确率, 用于监控模型的训练过程
        print('{} epoch, train loss: {:.4f}, train acc: {:.4f}'.format(i + 1, train_loss_all[-1], train_acc_all[-1]))
        print('{} epoch, val loss: {:.4f}, val acc: {:.4f}'.format(i + 1, val_loss_all[-1], val_acc_all[-1]))

        # 保存最佳模型
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            # 保存最佳模型的参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算每个 epoch 消耗的时间
        end_time = time.time()
        print('{} epoch consume time: {:.0f}m {:.0f}s'.format(i + 1, (end_time - start_time) // 60, (end_time - start_time) % 60))
        total_time += end_time - start_time
        start_time = end_time

    # 打印训练模型的总时间
    print('-' * 10)
    print('Total train time: {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))

    model.load_state_dict(best_model_wts)
    torch.save(model.load_state_dict(best_model_wts), './model/best_model.pth')

    # 先将训练过程的轮数以及每轮的训练集损失值、训练集准确率、验证集损失值、验证集准确率转换为 pandas 的 dataFrame 格式, 用于后续的分析和可视化
    train_process = pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'train_loss': train_loss_all,
        'train_acc': train_acc_all,
        'val_loss': val_loss_all,
        'val_acc': val_acc_all
    })
    # 把 dataFrame 保存到 CSV 文件中, 不包含索引列 （只有先转化为 dataFrame 才能保存到 CSV 文件中）
    train_process.to_csv('./train_parameter/train_parameter.csv', index=False)

    return train_process

# 绘制训练集和验证集的准确率和损失值的变化趋势
def matplot_acc_loss(train_process):
    # 设置图的大小
    plt.figure(figsize=(12, 4))

    # 绘制训练集和验证集的损失值的变化趋势
    plt.subplot(1, 2, 1) # 设置子图的位置，1 行 2 列，当前是第 1 个子图（第一列）
    plt.plot(train_process['epoch'], train_process['train_loss'], 'ro-', label='Train Loss')
    plt.plot(train_process['epoch'], train_process['val_loss'], 'bs-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制训练集和验证集的准确率的变化趋势
    plt.subplot(1, 2, 2) # 设置子图的位置，1 行 2 列，当前是第 2 个子图（第二列）
    plt.plot(train_process['epoch'], train_process['train_acc'], 'ro-',label='Train Accuracy')
    plt.plot(train_process['epoch'], train_process['val_acc'], 'bs-', label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # 显示并保存两张子图
    plt.savefig('./train_parameter/model_acc_loss.png')
    plt.show()

if __name__ == '__main__':
    train_dataloader, val_dataloader = deal_train_and_val_data()
    LeNetModel = LeNet()
    train_process = train_model(LeNetModel, train_dataloader, val_dataloader, 10)
    matplot_acc_loss(train_process)


