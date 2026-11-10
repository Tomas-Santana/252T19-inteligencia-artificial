import torch
from torch import nn

x = torch.rand(64, 1, 28, 28)  # 64 imágenes de 1 canal

conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
print("pesos:", conv.weight.shape)  # 8 filtros de 1 x 3 x 3
print("sesgos:", conv.bias.shape)  # un sesgo por filtro
print("salida:", conv(x).shape)
print("parámetros:", sum(p.numel() for p in conv.parameters()))
