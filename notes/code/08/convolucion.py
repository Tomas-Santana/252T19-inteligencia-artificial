import torch
from torch import nn

X = torch.tensor([[0., 1, 2, 2, 0],
                  [0, 0, 2, 1, 0],
                  [1, 0, 2, 2, 0],
                  [0, 0, 1, 2, 1],
                  [0, 1, 2, 2, 0]])
K = torch.tensor([[-1., 0, 1],
                  [-1, 0, 1],
                  [-1, 0, 1]])

conv = nn.Conv2d(1, 1, kernel_size=3, bias=False)
with torch.no_grad():
    conv.weight[0, 0] = K                 # el filtro, a mano
    out = conv(X.reshape(1, 1, 5, 5))  # (N, C, H, W)

print(out.shape)
print(out[0, 0])
