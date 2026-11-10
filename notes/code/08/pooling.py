import torch
from torch import nn

X = torch.tensor([[1., 3, 2, 0],
                  [4, 2, 1, 1],
                  [0, 1, 5, 2],
                  [2, 0, 3, 4]])
pool = nn.MaxPool2d(kernel_size=2)
print(pool(X.reshape(1, 1, 4, 4))[0, 0])

x = torch.rand(64, 8, 28, 28)
print(x.shape, "->", pool(x).shape)
print("parámetros:", sum(p.numel() for p in pool.parameters()))
