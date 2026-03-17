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
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)), # 全局平均池化层，将特征图的高宽都压缩为1
            nn.Flatten(),
            nn.Linear(1024, 10), # 10类别分类
        )

        # 初始化卷积层的权重, 因为GoogLeNet神经网络深度过深，偏导数在累乘过程中会导致梯度的指数级衰减；在这种情况下，如果初始参数随机值过大，会导致梯度爆炸或梯度消失问题
        # 这里用Kaiming初始化和正态分布初始化，它是一种针对ReLU激活函数的初始化方法，能够有效避免梯度消失问题
        # 它的基本思想是根据层的输入和输出维度，计算出一个合适的缩放因子，将权重初始化为符合正态分布的随机值
        # 这里的nonlinearity='relu' 是因为GoogLeNet神经网络中使用的是ReLU激活函数
        for m in self.modules():
            # 如果是卷积层，就初始化权重和偏置
            if isinstance(m, nn.Conv2d):
                # 初始化卷积层的权重w，这里的nonlinearity='relu' 是因为GoogLeNet神经网络中使用的是ReLU激活函数
                # 这里的mode='fan_out' 是因为在GoogLeNet神经网络中，卷积层的输出通道数是卷积核的数量，而不是输入通道数
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoogLeNet(Inception).to(device)
    summary(model, input_size=(1, 224, 224))
