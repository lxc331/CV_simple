import torch
from torch import nn
from torchsummary import summary

# 定义LeNet模型, 包含5个卷积层和3个全连接层
# nn.Module 就是上层接口的实现类，即pytorch实现类
class LeNet(nn.Module):
    # 初始化模型的层
    def __init__(self):
        super(LeNet, self).__init__() # 调用父类/接口（nn.Module）的初始化方法，初始化父类的层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid() # 激活函数
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) # AvgPool2d 和 AvgPool1d 都是池化层，但是 AvgPool2d 是对 2 维数据进行池化，而 AvgPool1d 是对 1 维数据进行池化
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.f5 = nn.Linear(in_features=5 * 5 * 16, out_features=120) # 5 * 5 * 16 是卷积层2的输出特征图的大小，展开成 1 维向量，120 是全连接层5的输出神经元的数量
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.f7 = nn.Linear(in_features=84, out_features=10)

    # 定义前向传播的计算过程
    def forward(self,x):
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x

if __name__ == '__main__': # 用主函数测试模型的前向传播
    # 测试模型的前向传播
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 实例化模型
    model = LeNet().to(device) # 将模型移动到 GPU 上或者 CPU 上
    # 打印模型的结构, 用 summary 函数，它的作用是打印模型的结构，包括层的类型、参数数量、输出大小等
    print(summary(model, (1, 28, 28))) # 打印模型的结构，输入数据的大小为 (1, 28, 28)，即 1 通道的 28x28 图像




