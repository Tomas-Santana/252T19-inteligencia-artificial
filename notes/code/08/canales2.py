import torch
from torch import nn

x = torch.rand(64, 8, 14, 14)  # 8 canales de entrada

conv = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
print("pesos:", conv.weight.shape)  # 16 filtros de 8 x 3 x 3
print("salida:", conv(x).shape)
print("parámetros:", sum(p.numel() for p in conv.parameters()))
