import torch
from torch import nn

torch.manual_seed(0)
dropout = nn.Dropout(p=0.5)
x = torch.ones(1, 8)

dropout.train()          # entrenamiento: apaga unidades al azar
print(dropout(x))
dropout.eval()           # evaluación: no hace nada
print(dropout(x))
