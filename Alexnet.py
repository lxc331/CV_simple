import torch
import numpy as np
import pandas as pd

# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
