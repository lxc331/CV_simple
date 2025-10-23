import torch
# 检查 PyTorch 是否可用
if torch.cuda.is_available():
   device = "cuda"
else:
   device = "cpu"
# 创建一个张量（Tensor）
x = torch.rand(5, 3).to(device)
print(f"Created tensor on {device}:")
print(x)
# 测试简单运算
y = torch.rand(5, 3).to(device)
z = x + y
print("Result of adding two tensors:")
print(z)