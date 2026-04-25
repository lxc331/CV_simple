import torch
from torch import nn
from torchsummary import summary

class Residual(nn.Module):
    def __init__(self,in_channel,kernel_num,use_1conv=False,strides=1): # use_1conv 是否使用1*1卷积层
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=kernel_num, kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(in_channels=kernel_num, out_channels=kernel_num, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(kernel_num) # 归一化层, 对卷积层的输出进行归一化,kernel_num为通道数,对每个通道进行归一化
        self.bn2 = nn.BatchNorm2d(kernel_num)
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=in_channel,out_channels=kernel_num,kernel_size=1,stride=strides,padding=0)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.ReLU(self.bn2(self.conv2(y)))
        if self.conv3 is not None:
            x = self.conv3(x)
        y = self.ReLU(y + x)
        return y

class ResNet18(nn.Module):
    def __init__(self, Residual):
        super(ResNet18, self).__init__()
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64), # 归一化层, 对卷积层的输出进行归一化,kernel_num为通道数,对每个通道进行归一化
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.Block2 = nn.Sequential(
            Residual(in_channel=64, kernel_num=64, use_1conv=False, strides=1),
            Residual(in_channel=64, kernel_num=64, use_1conv=False, strides=1),
        )
        self.Block3 = nn.Sequential(
            Residual(in_channel=64, kernel_num=128, use_1conv=True, strides=2),
            Residual(in_channel=128, kernel_num=128, use_1conv=False, strides=1),
        )
        self.Block4 = nn.Sequential(
            Residual(in_channel=128, kernel_num=256, use_1conv=True, strides=2),
            Residual(in_channel=256, kernel_num=256, use_1conv=False, strides=1),
        )
        self.Block5 = nn.Sequential(
            Residual(in_channel=256, kernel_num=512, use_1conv=True, strides=2),
            Residual(in_channel=512, kernel_num=512, use_1conv=False, strides=1),
        )
        self.Block6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # 全局平均池化层，将特征图的高宽都压缩为1
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=10),
        )
        # 注意一件事，ResNet18 可以不用参数初始化，直接使用即可，因为参数初始化在 forward 中进行的
        # 因为ResNet18 模型的BN层在 forward 中进行参数初始化，所以不需要在 __init__ 中进行参数初始化

    def forward(self, x):
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        x = self.Block4(x)
        x = self.Block5(x)
        x = self.Block6(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18(Residual).to(device)
    summary(model, input_size=(1, 224, 224))

