import torch
from cnn import MyCNN

model = MyCNN()
x = torch.rand(1, 1, 28, 28)  # una imagen inventada
print("entrada", tuple(x.shape))
x = torch.relu(model.conv1(x))
print("conv1  ", tuple(x.shape))
x = model.pool(x)
print("pool   ", tuple(x.shape))
x = torch.relu(model.conv2(x))
print("conv2  ", tuple(x.shape))
x = model.pool(x)
print("pool   ", tuple(x.shape))
x = x.flatten(start_dim=1)
print("flatten", tuple(x.shape))
x = model.out(x)
print("densa  ", tuple(x.shape))

for nombre, p in model.named_parameters():
    print(f"{nombre:13} {str(tuple(p.shape)):15} {p.numel():5}")
