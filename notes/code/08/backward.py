import torch

x, y = 2.0, 10.0
w1 = torch.tensor(1.0, requires_grad=True)
w2 = torch.tensor(1.0, requires_grad=True)

z = w1 * x                 # forward pass
a = torch.relu(z)
y_hat = w2 * a
L = (y_hat - y) ** 2

L.backward()               # backpropagation
print("L =", L.item())
print("dL/dw1 =", w1.grad.item())
print("dL/dw2 =", w2.grad.item())
