import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

# 定义 AlexNet 模型, 包含5个卷积层和3个全连接层
# nn.Module 就是上层接口的实现类，即pytorch实现类
class AlexNet(nn.Module):
    # 初始化模型的层（构造函数）, 定义模型的层和激活函数，给成员变量赋初值
    def __init__(self):
        super(AlexNet, self).__init__() # 调用父类/接口（nn.Module）的初始化方法，初始化父类的层

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(in_features=6 * 6 * 256, out_features=4096)
        self.f2 = nn.Linear(in_features=4096, out_features=4096)
        self.f3 = nn.Linear(in_features=4096, out_features=10)

    # 前向传播函数，定义模型的前向传播过程，即层的连接顺序
    def forward(self, x):
        # 简要嵌套的写法，将卷积层、激活函数和池化层嵌套在一起，避免太长
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.relu(self.f1(x))
        # 全连接层后添加dropout层，防止过拟合
        x = F.dropout(x,0.5)
        x = self.relu(self.f2(x))
        # 全连接层后添加dropout层，防止过拟合
        x = F.dropout(x,0.5)
        x = self.f3(x)

        return x

if __name__ == '__main__':
    # 测试模型的前向传播
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)

    summary(model, (1, 227, 227))
