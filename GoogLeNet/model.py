import torch
from torch import nn
from torchsummary import summary

class Inception(nn.Module):
    def __init__(self, in_channels, kernel_num1, kernel_num2, kernel_num3,kernel_num4):
        super(Inception,self).__init__() # 调用父类的构造函数,初始化父类的属性
        self.ReLU = nn.ReLU() # 激活函数

        # 卷积层步幅为1不用写，默认就是1；填充则不是0都要写，因为默认填充为0
        # 卷积层输出通道数大小实际上取决于卷积核的数量，且与其相等
        # 路线1，1x1卷积层
        self.path1 = nn.Conv2d(in_channels=in_channels, out_channels=kernel_num1, kernel_size=1)

        # 路线2，1x1卷积层后接3x3卷积层
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=kernel_num2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=kernel_num2[0], out_channels=kernel_num2[1], kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 路线3，1x1卷积层后接5x5卷积层
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=kernel_num3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=kernel_num3[0], out_channels=kernel_num3[1], kernel_size=5, padding=2),
            nn.ReLU()
        )

        # 路线4，3x3最大池化层后接1x1卷积层
        self.path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1), # 注意池化层步长为1，必须要写，因为池化层步幅默认为感受野大小
            nn.Conv2d(in_channels=in_channels, out_channels=kernel_num4, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.ReLU(self.path1(x))
        x2 = self.path2(x)
        x3 = self.path3(x)
        x4 = self.path4(x)
        return torch.cat([x1, x2, x3, x4], dim=1) # 通道数维度拼接,dim=1表示通道数维度

class GoogLeNet(nn.Module):
    def __init__(self, Inception):
        super(GoogLeNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3), # 由于训练数据集是灰度图，所以输入通道数为1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Inception模块串接
        self.block3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
        )
