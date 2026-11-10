import torch
from torch import nn


class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.out = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # (N, 8, 14, 14)
        x = self.pool(torch.relu(self.conv2(x)))   # (N, 16, 7, 7)
        x = x.flatten(start_dim=1)                 # (N, 784)
        return self.out(x)                       # (N, 10)
