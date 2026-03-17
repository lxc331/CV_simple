# PytorchCNN 项目简介

## 项目概述
PytorchCNN 是一个基于 PyTorch 框架构建的卷积神经网络（CNN）实验项目，旨在实现和比较多种经典的 CNN 架构，包括 LeNet、AlexNet、VGG16 和 GoogLeNet。该项目使用 Fashion-MNIST 数据集进行模型训练和测试。

## 项目结构
```
PytorchCNN/
├── AlexNet/           # AlexNet 模型实现
├── GoogLeNet/         # GoogLeNet 模型实现  
├── LeNet/             # LeNet 模型实现
├── VGG16/             # VGG16 模型实现
├── pytorch_test.py    # PyTorch 测试脚本
├── readme.md          # 项目说明文档
└── real_image_database_process.py  # 实际图像数据库处理脚本
```

## 各子项目特点

### AlexNet
- 实现经典的 AlexNet 架构
- 包含卷积层、池化层、全连接层等组件
- 使用 ReLU 激活函数和 Dropout 正则化

### GoogLeNet
- 实现 GoogLeNet 架构及其核心的 Inception 模块
- 采用多分支并行结构，提高计算效率
- 在保持精度的同时减少参数数量

### LeNet
- 实现最经典的 LeNet-5 架构
- 适合初学者理解 CNN 基本概念
- 结构简单但功能完整

### VGG16
- 实现 VGG16 深度卷积网络
- 使用多个小型卷积核堆叠的方式构建深层网络
- 体现了"深度"对于网络性能的重要性

## 主要功能
- **数据处理**：加载和预处理 Fashion-MNIST 数据集
- **模型训练**：支持 CPU/GPU 训练，可调节批次大小、学习率等超参数
- **模型验证**：实时监控训练和验证准确率与损失
- **结果可视化**：绘制训练过程中的准确率和损失曲线
- **模型保存**：自动保存最佳模型权重

## 技术栈
- **框架**：PyTorch
- **数据集**：Fashion-MNIST
- **可视化**：matplotlib
- **数据处理**：pandas, numpy

## 应用场景
- 深度学习教学与研究
- CNN 架构对比实验
- 图像分类算法验证
- 模型性能基准测试

这个项目为学习和理解经典 CNN 架构提供了一个完整的实践平台，特别适合深度学习初学者掌握卷积神经网络的基本原理和实现技巧。
        

## 附录
### 安装命令
#### 清华源
- pip install 包名 -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.douban.com

#### 豆瓣源
- pip install opencv-python==4.3.0.38 -i https://pypi.douban.com/simple --trusted-host pypi.douban.com

---

### 一些注意：
### 为什么我在模型搭建的时候，最后不需要添加 softmax 激活函数？
#### 原因
- 1.损失函数已包含softmax计算：在model_train.py中，训练过程使用的是nn.CrossEntropyLoss()作为损失函数，这个损失函数内部已经集成了log_softmax和nll_loss的计算。如果在模型中再添加softmax，会导致重复计算。

- 2.推理时的处理方式：在模型推理阶段，我们通常需要得到每个类别的概率分布。通过对模型输出应用softmax函数，我们可以将原始的logits转换为概率值。然而，由于softmax是一个归一化操作，它会改变原始的logits值的相对大小关系。而在分类任务中，我们更关注的是类别之间的相对差异，而不是绝对的概率值。因此，在推理阶段，我们通常直接使用模型的原始输出（logits）进行预测，而不需要对其应用softmax。

- 3.PyTorch的最佳实践：对于分类任务，PyTorch通常建议在模型的最后一层不添加softmax激活函数，而是让损失函数处理这一步骤。这可以提高数值稳定性并简化代码结构。

- 结论：当前LeNet模型的forward函数实现是合理的，不需要额外添加softmax激活函数。如果要获取概率分布，可以在推理阶段对输出应用softmax。