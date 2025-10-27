
### 安装命令
#### 清华源
- pip install 包名 -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.douban.com

#### 豆瓣源
- pip install opencv-python==4.3.0.38 -i https://pypi.douban.com/simple --trusted-host pypi.douban.com

---

AlexNet总结

1、AlexNet和LeNet设计上有一脉相承之处，也有区别。为了适应ImageNet中大尺寸图像，其使用了大尺寸的卷积核，从11x11到5x5到3x3，AlexNet的卷积层和全连接层也带来更多的参数6000万，这一全新设计的CNN结构在图像分类领域取大幅超越传统机器学习，自此之后CNN在图像分类领域被广泛应用。

2、使用了Relu替换了传统的sigmoid或tanh作为激活函数，大大加快了收敛，减少了模型训练耗时。

3、使用了Dropout，提高了模型的准确度，减少过拟合，这一方
式再后来也被广泛采用。

4、在CNN中使用重叠的最大池化。此前CNN中普遍使用平均池化AlexNet全部使用最大池化，避免平均池化的模糊化效果。并且AlexNet中提出让步长比池化核的尺寸小，这样池化层的输出之间会有重叠和覆盖，提升了特征的丰富性。

5、使用数据了2种数据扩增技术，大幅增加了训练数据，增加模型鲁棒性，减少过拟合。

6、使用了LRN正则化、多GPU并行训练的模式(不过之后并没有被广泛应用)