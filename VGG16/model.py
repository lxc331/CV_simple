import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__() # 调用父类/接口（nn.Module）的初始化方法，初始化父类的层
        # 第一个卷积块, nn.Sequential 是一个有序的容器，将多个层按**顺序**组合起来
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第二个卷积块
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第三个卷积块
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第四个卷积块
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第五个卷积块
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第六个卷积块, 平展+全连接层
        self.block6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=7 * 7 * 512, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=10),
            # 这里不用加softmax，因为后面会用cross entropy loss(交叉熵误差)，它会自动加上softmax
            # nn.Softmax(dim=1) # 对最后一层的输出进行softmax归一化，将输出转换为概率分布
        )

        # 初始化卷积层的权重, 因为VGG神经网络深度过深，偏导数在累乘过程中会导致梯度的指数级衰减；在这种情况下，如果初始参数随机值过大，会导致梯度爆炸或梯度消失问题
        # 这里用Kaiming初始化和正态分布初始化，它是一种针对ReLU激活函数的初始化方法，能够有效避免梯度消失问题
        # 它的基本思想是根据层的输入和输出维度，计算出一个合适的缩放因子，将权重初始化为符合正态分布的随机值
        # 这里的nonlinearity='relu' 是因为VGG神经网络中使用的是ReLU激活函数
        for m in self.modules():
            # 如果是卷积层，就初始化权重和偏置
            if isinstance(m, nn.Conv2d):
                # 初始化卷积层的权重w，这里的nonlinearity='relu' 是因为VGG神经网络中使用的是ReLU激活函数
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                # 如果有偏置项(y = wx + b 中的b)，就将偏置初始化为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 如果是全连接层，就初始化权重和偏置
            elif isinstance(m, nn.Linear):
                # 初始化全连接层的权重w，这里的mean=0, std=0.001 意思是将权重初始化为符合正态分布的随机值，均值为0，标准差为0.001
                nn.init.normal_(m.weight, mean=0, std=0.001)
                # 如果有偏置项(y = wx + b 中的b)，就将偏置初始化为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 注意：即使权重初始化了，也与可能最后权重w不收敛，这是因为优化算法（如梯度下降）在训练过程中会根据损失函数的梯度来更新参数，而初始参数的选择对最终结果有影响
        # 训练时，权重参数的更新是按批次更新的，不同的图像更新效果会有差异
        # 不同的初始参数选择可能导致不同的收敛结果，因此，需要根据具体情况进行调试和调整（一般是需要调整学习率、批量大小(>20(24左右)效果比较好)、迭代次数等超参数），以确保模型能够收敛到一个好的解
        # 实在不行就直接对最后的全连接层的神经元数量进行修改(改小)，防止调高批量大小时，批量大小过大会导致显存不足问题

    # 前向传播函数，定义模型的前向传播过程，即层的连接顺序
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        return x

if __name__ == '__main__':
    # 测试模型的前向传播
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VGG16().to(device)

    # summary(model, (1, 224, 224))
