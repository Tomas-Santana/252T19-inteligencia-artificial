import torch
from torch import nn


class MyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 128)
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        x = x.flatten(start_dim=1)  # (N, 1, 28, 28) -> (N, 784)
        x = torch.relu(self.hidden(x))  # (N, 128)
        return self.out(x)  # (N, 10): logits, sin softmax


if __name__ == "__main__":
    model = MyMLP()
    x = torch.rand(64, 1, 28, 28)  # un mini-batch de 64 imágenes
    print(model(x).shape)
    print(sum(p.numel() for p in model.parameters()))
