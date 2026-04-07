import torch
import numpy

print("torch version:", torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.is_available())
