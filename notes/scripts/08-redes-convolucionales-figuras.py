"""Genera los datos de las figuras y las salidas de código de la clase 8.

Uso: python notes/scripts/08-redes-convolucionales-figuras.py

1. Ejecuta cada fragmento de notes/codigo/08/ y guarda lo que imprime en
   notes/figs/08/salidas.json. Las notas y las diapositivas muestran el código
   y su salida real a partir de esos archivos.
2. Entrena MyMLP y MyCNN (definidas en notes/codigo/08/) sobre MNIST con la
   función `entrenar` de notes/codigo/08/entrenamiento.py, y guarda los
   resultados, los filtros aprendidos y los datos de las demás figuras en
   notes/figs/08/mnist.json.

La primera vez descarga MNIST (unos 11 MB) en notes/scripts/.cache/mnist.
"""

import contextlib
import gzip
import io
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

NOTAS = Path(__file__).parent.parent
CODIGO = NOTAS / "codigo/08"
FIGS = NOTAS / "figs/08"
FIGS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODIGO))

from cnn import MyCNN  # noqa: E402
from entrenamiento import entrenar, exactitud  # noqa: E402
from mlp import MyMLP  # noqa: E402


def limpiar(x):
    if isinstance(x, dict):
        return {k: limpiar(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [limpiar(v) for v in x]
    if isinstance(x, np.ndarray):
        return limpiar(x.tolist())
    if isinstance(x, (np.floating, float)):
        return round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    return x


def guardar(nombre, datos):
    with open(FIGS / f"{nombre}.json", "w") as f:
        json.dump(limpiar(datos), f, ensure_ascii=False)
    print(f"figs/08/{nombre}.json")


def a_imagen(m, simetrica=False):
    """Convierte una matriz en píxeles 0-255 (0 = blanco) para Typst.
    Con simetrica=True, el 0 queda en gris medio (para filtros con valores negativos)."""
    m = np.asarray(m, dtype=float)
    if simetrica:
        r = np.abs(m).max() or 1.0
        return np.round(127.5 + 127.5 * m / r).astype(int)
    r = m.max() or 1.0
    return np.round(255 * np.clip(m, 0, None) / r).astype(int)


# ---------------------------------------------------------------------------
# 1. Salidas de los fragmentos de código
# ---------------------------------------------------------------------------
fragmentos = ["backward", "mlp", "convolucion", "canales", "canales2", "padding", "pooling", "formas", "dropout"]
salidas = {}
for nombre in fragmentos:
    r = subprocess.run([sys.executable, f"{nombre}.py"], cwd=CODIGO, capture_output=True, text=True, check=True)
    salidas[nombre] = r.stdout.rstrip("\n")

# ---------------------------------------------------------------------------
# 2. MNIST
# ---------------------------------------------------------------------------
CACHE = Path(__file__).parent / ".cache/mnist"
URL = "https://raw.githubusercontent.com/fgnt/mnist/master/"


def cargar(nombre, offset):
    ruta = CACHE / nombre
    if not ruta.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"descargando {nombre}")
        urllib.request.urlretrieve(URL + nombre, ruta)
    return np.frombuffer(gzip.open(ruta).read(), np.uint8, offset=offset)


X_todo = cargar("train-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28)
y_todo = cargar("train-labels-idx1-ubyte.gz", 8)
X_test_np = cargar("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28)
y_test_np = cargar("t10k-labels-idx1-ubyte.gz", 8)


def tensor(X, y):
    return torch.tensor(X / 255, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


rng = np.random.default_rng(0)
orden = rng.permutation(len(y_todo))
i_train, i_val = orden[:50000], orden[50000:]
X_train, y_train = tensor(X_todo[i_train], y_todo[i_train])
X_val, y_val = tensor(X_todo[i_val], y_todo[i_val])
X_test, y_test = tensor(X_test_np, y_test_np)

resultados = {}
modelos = {}
for nombre, clase in [("mlp", MyMLP), ("cnn", MyCNN)]:
    torch.manual_seed(0)
    modelo = clase()
    texto = io.StringIO()
    inicio = time.time()
    with contextlib.redirect_stdout(texto):
        entrenar(modelo, X_train, y_train, X_val, y_val, epocas=3, lr=0.1)
    segundos = time.time() - inicio
    salidas[f"entrenar-{nombre}"] = texto.getvalue().rstrip("\n")
    resultados[nombre] = {
        "parametros": sum(p.numel() for p in modelo.parameters()),
        "validacion": exactitud(modelo, X_val, y_val),
        "prueba": exactitud(modelo, X_test, y_test),
        "segundos": segundos,
    }
    modelos[nombre] = modelo
    print(nombre, resultados[nombre])

guardar("salidas", salidas)

# Filtros aprendidos por la primera capa de la CNN y sus mapas de activación
cnn = modelos["cnn"]
filtros = cnn.conv1.weight.detach().numpy()[:, 0]  # (8, 3, 3)
ejemplo = 0
while y_test_np[ejemplo] != 7:
    ejemplo += 1
with torch.no_grad():
    mapas = torch.relu(cnn.conv1(X_test[ejemplo : ejemplo + 1]))[0].numpy()  # (8, 28, 28)

# ---------------------------------------------------------------------------
# 3. Filtros diseñados a mano sobre un dígito real
# ---------------------------------------------------------------------------
K_vertical = torch.tensor([[-1.0, 0, 1], [-1, 0, 1], [-1, 0, 1]])
K_horizontal = K_vertical.T.contiguous() * -1  # filas: arriba +1, abajo -1 -> detecta bordes horizontales
digito = X_test[ejemplo : ejemplo + 1]
with torch.no_grad():
    bordes_v = F.relu(F.conv2d(digito, K_vertical.reshape(1, 1, 3, 3), padding=1))[0, 0].numpy()
    bordes_h = F.relu(F.conv2d(digito, K_horizontal.reshape(1, 1, 3, 3), padding=1))[0, 0].numpy()

# ---------------------------------------------------------------------------
# 4. Un dígito desplazado: para un MLP es una entrada muy distinta
# ---------------------------------------------------------------------------
cuatro = int(np.flatnonzero(y_todo == 4)[3])
original = X_todo[cuatro, 0].astype(float)
desplazado = np.zeros_like(original)
DESPLAZAMIENTO = 4
desplazado[:, DESPLAZAMIENTO:] = original[:, :-DESPLAZAMIENTO]
d_desplazado = np.linalg.norm(original - desplazado)
otros = np.flatnonzero(y_todo != 4)
distancias = np.linalg.norm(X_todo[otros, 0].reshape(len(otros), -1).astype(float) - original.ravel(), axis=1)
cercano = int(otros[np.argmin(distancias)])
cambian = int(((original > 64) != (desplazado > 64)).sum())

# ---------------------------------------------------------------------------
# 5. Sobreajuste: una red grande con pocos datos, con y sin dropout
# ---------------------------------------------------------------------------


class RedGrande(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.oculta1 = nn.Linear(784, 512)
        self.oculta2 = nn.Linear(512, 512)
        self.salida = nn.Linear(512, 10)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.dropout(torch.relu(self.oculta1(x)))
        x = self.dropout(torch.relu(self.oculta2(x)))
        return self.salida(x)


def curvas(p, n=1000, epocas=150):
    torch.manual_seed(0)
    modelo = RedGrande(p)
    perdida = nn.CrossEntropyLoss()
    optimizador = torch.optim.SGD(modelo.parameters(), lr=0.05)
    Xs, ys = X_train[:n], y_train[:n]
    tr, va = [], []
    for epoca in range(epocas):
        modelo.train()
        idx = torch.randperm(n)
        for k in range(0, n, 64):
            lote = idx[k : k + 64]
            optimizador.zero_grad()
            perdida(modelo(Xs[lote]), ys[lote]).backward()
            optimizador.step()
        modelo.eval()
        with torch.no_grad():
            tr.append((epoca + 1, perdida(modelo(Xs), ys).item()))
            va.append((epoca + 1, perdida(modelo(X_val), y_val).item()))
    return {"train": tr, "val": va, "exactitud_val": exactitud(modelo, X_val, y_val)}


sobreajuste = {"sin": curvas(0.0), "con": curvas(0.5)}
for k, v in sobreajuste.items():
    mejor = min(v["val"], key=lambda t: t[1])
    print(f"sobreajuste {k} dropout: mínimo de validación en la época {mejor[0]}, exactitud {v['exactitud_val']:.3f}")

guardar(
    "mnist",
    {
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "resultados": resultados,
        "filtros": [a_imagen(f, simetrica=True) for f in filtros],
        "filtros_valores": filtros,
        "ejemplo": a_imagen(X_test_np[ejemplo, 0]),
        "mapas": [a_imagen(m) for m in mapas],
        "bordes_v": a_imagen(bordes_v),
        "bordes_h": a_imagen(bordes_h),
        "original": original.astype(int),
        "desplazado": desplazado.astype(int),
        "desplazamiento": DESPLAZAMIENTO,
        "cercano": X_todo[cercano, 0],
        "cercano_clase": y_todo[cercano],
        "d_desplazado": d_desplazado,
        "d_cercano": float(distancias.min()),
        "cambian": cambian,
        "sobreajuste": sobreajuste,
    },
)
