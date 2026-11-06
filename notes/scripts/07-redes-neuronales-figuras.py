"""Genera los datos de las figuras de la clase 7 (redes neuronales).

Uso: python notes/scripts/07-redes-neuronales-figuras.py

Escribe un archivo JSON por figura en notes/figs/07/. Las notas
(07-redes-neuronales.typ) los leen con json(...) y dibujan las figuras con cetz.
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

FIGS = Path(__file__).parent.parent / "figs/07"
FIGS.mkdir(parents=True, exist_ok=True)


def guardar(nombre, datos):
    def limpiar(x):
        if isinstance(x, dict):
            return {k: limpiar(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [limpiar(v) for v in x]
        if isinstance(x, (np.floating, float)):
            return round(float(x), 4)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.ndarray):
            return limpiar(x.tolist())
        return x

    with open(FIGS / f"{nombre}.json", "w") as f:
        json.dump(limpiar(datos), f)
    print(f"figs/07/{nombre}.json")


def frontera(predecir_proba, limites, n=300):
    """Curvas de nivel p = 0.5 de un clasificador, como lista de polilíneas."""
    x0, x1, y0, y1 = limites
    xs, ys = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    Z = predecir_proba(np.c_[xs.ravel(), ys.ravel()]).reshape(xs.shape)
    cs = plt.contour(xs, ys, Z, levels=[0.5])
    segmentos = [s[::3].tolist() + [s[-1].tolist()] for s in cs.allsegs[0] if len(s) > 5]
    plt.close("all")
    return segmentos


# ---------------------------------------------------------------------------
# Lunas: regresión logística vs. redes con una capa oculta
# ---------------------------------------------------------------------------
X, y = make_moons(n_samples=200, noise=0.2, random_state=0)
X_te, y_te = make_moons(n_samples=2000, noise=0.2, random_state=1)
limites = (-1.6, 2.6, -1.2, 1.7)

paneles = {}
log = LogisticRegression().fit(X, y)
paneles["logistica"] = {
    "frontera": frontera(lambda P: log.predict_proba(P)[:, 1], limites),
    "exactitud": log.score(X_te, y_te),
}
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    for nombre, ocultas in [("h3", 3), ("h32", 32)]:
        red = MLPClassifier(
            hidden_layer_sizes=(ocultas,), activation="relu", solver="lbfgs", max_iter=3000, random_state=0
        ).fit(X, y)
        paneles[nombre] = {
            "frontera": frontera(lambda P: red.predict_proba(P)[:, 1], limites),
            "exactitud": red.score(X_te, y_te),
            "parametros": sum(w.size for w in red.coefs_) + sum(b.size for b in red.intercepts_),
        }

guardar(
    "lunas",
    {
        "neg": X[y == 0],
        "pos": X[y == 1],
        "limites": limites,
        "paneles": paneles,
    },
)

# ---------------------------------------------------------------------------
# MNIST: dígitos reales y la red 784-16-16-10 (13002 parámetros) de las notas
# ---------------------------------------------------------------------------
import gzip
import urllib.request

import torch
from torch import nn

CACHE = Path(__file__).parent / ".cache/mnist"
URL = "https://raw.githubusercontent.com/fgnt/mnist/master/"


def cargar(nombre, offset):
    ruta = CACHE / nombre
    if not ruta.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"descargando {nombre}")
        urllib.request.urlretrieve(URL + nombre, ruta)
    return np.frombuffer(gzip.open(ruta).read(), np.uint8, offset=offset)


X_tr = cargar("train-images-idx3-ubyte.gz", 16).reshape(-1, 784)
y_tr = cargar("train-labels-idx1-ubyte.gz", 8)
X_te = cargar("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 784)
y_te = cargar("t10k-labels-idx1-ubyte.gz", 8)

# Tres ejemplos de entrenamiento de cada dígito
muestras = [[X_tr[np.flatnonzero(y_tr == d)[k]] for d in range(10)] for k in range(3)]

torch.manual_seed(0)
red = nn.Sequential(nn.Linear(784, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 10))
n_parametros = sum(p.numel() for p in red.parameters())
perdida = nn.CrossEntropyLoss()
optimizador = torch.optim.SGD(red.parameters(), lr=0.1)

Xt = torch.tensor(X_tr / 255, dtype=torch.float32)
yt = torch.tensor(y_tr, dtype=torch.long)
EPOCAS, LOTE, CADA = 5, 64, 100
curva, acumulado, paso = [], 0.0, 0
for epoca in range(EPOCAS):
    orden = torch.randperm(len(yt))
    for k in range(0, len(yt), LOTE):
        idx = orden[k : k + LOTE]
        optimizador.zero_grad()
        L = perdida(red(Xt[idx]), yt[idx])
        L.backward()
        optimizador.step()
        acumulado += L.item()
        paso += 1
        if paso % CADA == 0:
            curva.append((paso, acumulado / CADA))
            acumulado = 0.0

with torch.no_grad():
    probs = torch.softmax(red(torch.tensor(X_te / 255, dtype=torch.float32)), dim=1).numpy()
pred = probs.argmax(axis=1)
exactitud = (pred == y_te).mean()
print(f"MNIST: {n_parametros} parámetros, exactitud de prueba {exactitud:.4f}, {paso} pasos")

rng = np.random.default_rng(1)
bien = rng.choice(np.flatnonzero(pred == y_te), 12, replace=False)
mal = rng.choice(np.flatnonzero(pred != y_te), 6, replace=False)


def ejemplo(i):
    return {"pixeles": X_te[i], "real": y_te[i], "pred": pred[i], "prob": probs[i, pred[i]]}


guardar(
    "mnist",
    {
        "muestras": muestras,
        "parametros": n_parametros,
        "m_train": len(y_tr),
        "m_test": len(y_te),
        "exactitud": exactitud,
        "errores": int((pred != y_te).sum()),
        "epocas": EPOCAS,
        "lote": LOTE,
        "pasos": paso,
        "curva": curva,
        "bien": [ejemplo(i) for i in bien],
        "mal": [ejemplo(i) for i in mal],
    },
)
