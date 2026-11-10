import torch
from torch import nn

x = torch.rand(1, 1, 28, 28)
for padding, stride in [(0, 1), (1, 1), (1, 2)]:
    conv = nn.Conv2d(1, 8, kernel_size=3, padding=padding, stride=stride)
    print(f"padding={padding}, stride={stride}:", tuple(conv(x).shape))
