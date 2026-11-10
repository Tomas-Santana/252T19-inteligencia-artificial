import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def accuracy(model, X, y):
    model.eval()              # modo evaluación
    with torch.no_grad():      # sin registrar operaciones
        predicted_classes = model(X).argmax(dim=1)
    return (predicted_classes == y).float().mean().item()


def train(model, X_train, y_train, X_val, y_val, num_epochs=3, lr=0.1):
    datos = DataLoader(TensorDataset(X_train, y_train),
                       batch_size=64, shuffle=True)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoca in range(num_epochs):
        model.train()                # modo entrenamiento
        for X_batch, y_batch in datos:  # un paso por mini-batch
            optimizer.zero_grad()
            L = loss(model(X_batch), y_batch)
            L.backward()
            optimizer.step()
        acc = accuracy(model, X_val, y_val)
        print(f"época {epoca + 1}: validación {acc:.3f}")
