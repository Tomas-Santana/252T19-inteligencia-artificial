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
# Curva de pérdida al entrenar una red sobre las lunas (descenso por gradiente)
# ---------------------------------------------------------------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    red = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        solver="sgd",
        learning_rate_init=0.1,
        momentum=0.0,
        batch_size=200,
        max_iter=1000,
        n_iter_no_change=1000,
        random_state=0,
    ).fit(X, y)
guardar("perdida", {"perdida": list(enumerate(red.loss_curve_, start=1)), "exactitud": red.score(X_te, y_te)})
