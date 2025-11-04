
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