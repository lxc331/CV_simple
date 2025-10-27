from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt


# 加载FashionMNIST数据集, root 是数据集的根目录，train=True 表示加载训练集，download=True 表示如果数据集不存在，就从网上下载,
# transform 是对数据进行预处理的操作，这里是将原本28x28的图像 resize 到 224x224 大小，然后转换为张量，为了画图看的更清楚
# 这里的 transforms.Compose 是将多个预处理操作组合起来，这里是将 Resize 和 ToTensor 操作组合起来，先将图像 resize 到 224x224 大小，然后转换为张量
train_data = FashionMNIST(root='./data',
                          train=True,
                          download=True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]))

# 定义训练集的 DataLoader,加载训练集, batch_size 是每个批次的样本数量，shuffle=True 表示在每个 epoch 开始前，打乱样本的顺序，num_workers 是加载数据的线程数量
# train_loader 是一个四维张量，形状为 (64, 1, 224, 224)，表示 64 个样本，每个样本是1个通道的 224x224 的灰度图像，每个像素的值在 [0, 1] 之间
# FashionMNIST是灰度图像数据集，因此通道数为1
train_loader = data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)

# 遍历训练集的 DataLoader, 每个批次的样本数量是 64 个
# 每个批次的样本数量是 64 个，每个样本是一个 224x224 的灰度图像，每个像素的值在 [0, 1] 之间
# 每个样本的类别标签是一个整数，范围是 [0, 9]，表示该样本所属的类别
for step, (b_x,b_y) in enumerate(train_loader): # step 是当前批次的索引，b_x 是当前批次的样本，b_y 是当前批次的类别标签
    if step > 0: # 只遍历第一个批次
        break
    batch_x = b_x.squeeze().numpy() # 从四维张量 64x1x224x224 中移除张量中大小为1的维度，然后转换为 numpy 数组（方便可视化），得到 64x224x224 的数组
    batch_y = b_y.numpy() # 将类别标签转换为 numpy 数组（方便可视化）
    class_label = train_data.classes # 获取训练集的类别标签
    print(batch_x.shape, batch_y.shape)
    print(batch_y)

# 可视化第一个批次的样本
plt.figure(figsize=(12,5)) # 12x5 的子图，每个子图的大小为 12x5 英寸
for i in np.arange(len(batch_y)): # 遍历第一个批次的样本
    plt.subplot(4,16,i + 1) # 8行8列，第i+1个子图
    plt.imshow(batch_x[i, : , : ], cmap=plt.cm.gray) # 显示第i个样本的灰度图像
    plt.title(class_label[batch_y[i]],size=10) # 显示第i个样本的类别标签
    plt.axis('off') # 关闭坐标轴
    plt.subplots_adjust(wspace=0.05) # 调整子图之间的水平间距为0.05
plt.show()

