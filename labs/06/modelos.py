"""Clase 8, laboratorio: las redes neuronales.

Este es el ÚNICO archivo que hay que editar. El notebook lab.ipynb importa
estas clases, las verifica y las entrena.

Cómo es una red en PyTorch:
  - __init__ DECLARA las capas que tienen parámetros, como atributos de self.
    Lo primero es llamar super().__init__().
  - forward DESCRIBE el cálculo: qué le pasa a x, en qué orden.
    Las operaciones sin parámetros (torch.relu, x.flatten) van solo aquí.
  - La red devuelve los scores (logits), SIN softmax: nn.CrossEntropyLoss lo aplica.

Las imágenes llegan como tensores de forma (N, 1, 28, 28):
N imágenes, 1 canal (escala de grises), 28 x 28 píxeles.
"""

import torch
from torch import nn


class MyMLP(nn.Module):
    """Parte 1. Red densa: 784 -> 128 -> 10."""

    def __init__(self):
        super().__init__()
        # CÓDIGO AQUÍ. Declarar dos capas densas (nn.Linear):
        #   self.hidden: de 784 entradas a 128 salidas
        #   self.output: de 128 entradas a 10 salidas (una por dígito)

    def forward(self, x):
        # CÓDIGO AQUÍ:
        #   1. aplanar cada imagen: (N, 1, 28, 28) -> (N, 784)
        x = x.flatten(start_dim=1)
        #   2. pasar por self.hidden y aplicar torch.relu
        #   3. pasar por self.output y devolver el resultado (sin softmax)
        raise NotImplementedError("Falta completar MyMLP.forward en modelos.py")


def train_step(model, X_batch, y_batch, loss_fn, optimizer):
    """Parte 1b. Un paso de descenso por gradiente sobre un mini-batch.

    Devuelve la pérdida del mini-batch como número (float).
    """
    # CÓDIGO AQUÍ:
    #   1. borrar los gradientes del paso anterior: optimizer.zero_grad()
    #   2. forward: calcular los scores de la red para X_batch
    #   3. calcular la pérdida con loss_fn(scores, y_batch)
    #   4. backward: calcular los gradientes de la pérdida
    #   5. dar el paso: optimizer.step()
    #   6. devolver la pérdida como float (con .item())
    raise NotImplementedError("Falta completar train_step en modelos.py")


class MyCNN(nn.Module):
    """Parte 2. Red convolucional:
    [conv 3x3 (8 filtros) -> ReLU -> max pool 2x2]
    [conv 3x3 (16 filtros) -> ReLU -> max pool 2x2]
    -> aplanar -> densa -> 10
    Las convoluciones usan padding=1 para no achicar la imagen.
    """

    def __init__(self):
        super().__init__()
        # CÓDIGO AQUÍ. Usar exactamente estos nombres (el notebook los usa):
        #   self.conv1: nn.Conv2d de 1 canal de entrada a 8 filtros, kernel_size=3, padding=1
        #   self.conv2: nn.Conv2d de 8 canales a 16 filtros, kernel_size=3, padding=1
        #   self.pool:  nn.MaxPool2d(kernel_size=2) (se puede usar dos veces: no tiene parámetros)
        #   self.dense: nn.Linear(???, 10)
        #       ¿Cuántas entradas recibe?

    def forward(self, x):
        # CÓDIGO AQUÍ:
        #   1. conv1 -> ReLU -> pool
        #   2. conv2 -> ReLU -> pool
        #   3. aplanar: x.flatten(start_dim=1)
        #   4. densa (sin softmax)
        raise NotImplementedError("Falta completar MyCNN.forward en modelos.py")


class MyCNNVariant(nn.Module):
    """Parte 3, para la casa.
    Copiar aquí MyCNN y experimentar con su arquitectura, por ejemplo:
      - el número de filtros (16 y 32 en lugar de 8 y 16)
      - el tamaño del filtro (kernel_size=5, con padding=2)
      - un tercer bloque de convolución
      - un paso de dropout
      - una capa densa oculta antes de la salida
    Usar también los nombres self.conv1 y self.dense para la primera convolución
    y la última capa densa.
    """

    def __init__(self):
        super().__init__()
        # CÓDIGO AQUÍ

    def forward(self, x):
        # CÓDIGO AQUÍ
        raise NotImplementedError("Falta completar MyCNNVariant en modelos.py (reto para la casa)")
