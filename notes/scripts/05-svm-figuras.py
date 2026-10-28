"""
Genera los datos de las figuras de las notas de máquinas de soporte vectorial
y kernels.

Cada figura se escribe en notes/figs/05/<nombre>.json y Typst la lee con json(...).
Se puede ejecutar desde cualquier carpeta:

    python notes/scripts/05-svm-figuras.py

También se puede importar (lo hace notebooks/05-svm.ipynb): al importarlo solo
se construyen los datos y los modelos; los JSON se escriben en main().

Convenciones (las mismas de las notas):
    y ∈ {-1, 1},  score = w^T x + b
    Margen duro:  min 1/2 ||w||^2  s.a.  y (w^T x + b) >= 1
    Margen suave: min 1/2 ||w||^2 + C Σ ξ_i
    En scikit-learn, SVC(C=...) resuelve exactamente el margen suave de las
    notas, y SVC(kernel="rbf", gamma=...) usa K(x, z) = exp(-gamma ||x - z||^2),
    es decir, gamma = 1 / (2 σ^2).
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.svm import SVC

FIGS = Path(__file__).resolve().parent.parent / "figs" / "05"
C_DURO = 1e6  # C muy grande: equivale al margen duro con datos separables


def guardar(nombre, datos):
    with open(FIGS / f"{nombre}.json", "w") as fh:
        json.dump(datos, fh, indent=1)


def redondear(arr, d=4):
    return np.round(np.asarray(arr, dtype=float), d).tolist()


def por_clase(X, y):
    return {"neg": redondear(X[y == -1]), "pos": redondear(X[y == 1])}


def margen_geometrico(w, b, X, y):
    return float(np.min(y * (X @ w + b)) / np.linalg.norm(w))


def svm_lineal(X, y, C=C_DURO):
    modelo = SVC(kernel="linear", C=C, tol=1e-8).fit(X, y)
    w = modelo.coef_.ravel()
    b = float(modelo.intercept_[0])
    return modelo, w, b


def resumen_lineal(modelo, w, b, X, y):
    return {
        "w": w.tolist(),
        "b": b,
        "sv": redondear(modelo.support_vectors_),
        "n_sv": int(len(modelo.support_)),
        "margen": margen_geometrico(w, b, X, y),
        "norma_w": float(np.linalg.norm(w)),
    }


def frontera_kernel(modelo, x_lim, y_lim, n=300):
    """Segmentos de la curva decision_function = 0 (vía matplotlib)."""
    xs = np.linspace(*x_lim, n)
    ys = np.linspace(*y_lim, n)
    XX, YY = np.meshgrid(xs, ys)
    Z = modelo.decision_function(np.column_stack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    fig, ax = plt.subplots()
    cs = ax.contour(XX, YY, Z, levels=[0.0])
    segmentos = [redondear(s, 3) for s in cs.allsegs[0] if len(s) > 2]
    plt.close(fig)
    return segmentos


# ===========================================================================
# 1. Datos separables: margen óptimo, confianza y margen geométrico
# ===========================================================================
rng = np.random.default_rng(3)
m1 = 14
X_neg = rng.normal([1.3, 1.2], 0.45, (m1, 2))
X_pos = rng.normal([3.5, 3.4], 0.45, (m1, 2))
X_sep = np.vstack([X_neg, X_pos])
y_sep = np.concatenate([-np.ones(m1), np.ones(m1)])

modelo_opt, w_opt, b_opt = svm_lineal(X_sep, y_sep)
assert np.all(y_sep * (X_sep @ w_opt + b_opt) >= 1 - 1e-4), "no separa con margen 1"

# Otra frontera que también separa, escogida a mano, con menor margen
w_otra = np.array([1.0, 0.35])
b_otra = -3.15
assert np.all(y_sep * (X_sep @ w_otra + b_otra) > 0), "la otra frontera no separa"
# La escalamos para que su margen funcional mínimo sea 1 (como en las notas)
escala = 1 / np.min(y_sep * (X_sep @ w_otra + b_otra))
w_otra, b_otra = w_otra * escala, b_otra * escala

# Margen geométrico de un ejemplo positivo (punto A) y su proyección B
idx_A = int(np.argmax(X_pos @ w_opt))  # el positivo más lejano, se ve mejor
A = X_pos[idx_A]
gamma_A = float((w_opt @ A + b_opt) / np.linalg.norm(w_opt))
B = A - gamma_A * w_opt / np.linalg.norm(w_opt)
assert abs(w_opt @ B + b_opt) < 1e-8

separables = {
    **por_clase(X_sep, y_sep),
    "opt": resumen_lineal(modelo_opt, w_opt, b_opt, X_sep, y_sep),
    "otra": {
        "w": w_otra.tolist(),
        "b": float(b_otra),
        "margen": margen_geometrico(w_otra, b_otra, X_sep, y_sep),
    },
    "A": A.tolist(),
    "B": B.tolist(),
    "gamma_A": gamma_A,
}


# ===========================================================================
# 2. Kernel gaussiano con tres valores de σ (datos en forma de medialuna)
# ===========================================================================
X_moon, y01 = make_moons(n_samples=120, noise=0.25, random_state=4)
y_moon = 2 * y01 - 1
X_test, y01_t = make_moons(n_samples=1000, noise=0.25, random_state=5)
y_test = 2 * y01_t - 1

x_lim, y_lim = (-1.6, 2.6), (-1.2, 1.7)
sigmas = {"grande": 2.0, "medio": 0.5, "pequeno": 0.1}
paneles = {}
for nombre, sigma in sigmas.items():
    gamma = 1 / (2 * sigma**2)
    modelo = SVC(kernel="rbf", C=10, gamma=gamma).fit(X_moon, y_moon)
    paneles[nombre] = {
        "sigma": sigma,
        "frontera": frontera_kernel(modelo, x_lim, y_lim),
        "sv": redondear(modelo.support_vectors_),
        "n_sv": int(len(modelo.support_)),
        "exactitud_train": float(modelo.score(X_moon, y_moon)),
        "exactitud_test": float(modelo.score(X_test, y_test)),
    }

gaussiano = {**por_clase(X_moon, y_moon), "paneles": paneles, "m": len(y_moon)}


# ===========================================================================
# 3. Un outlier con margen duro y con margen suave
# ===========================================================================
rng = np.random.default_rng(21)
m3 = 12
Xo_neg = rng.normal([1.4, 1.3], 0.4, (m3, 2))
Xo_pos = rng.normal([3.4, 3.3], 0.4, (m3, 2))
X_base = np.vstack([Xo_neg, Xo_pos])
y_base = np.concatenate([-np.ones(m3), np.ones(m3)])

outlier = np.array([2.35, 3.4])  # ejemplo negativo cerca de los positivos
X_out = np.vstack([X_base, outlier])
y_out = np.concatenate([y_base, [-1.0]])

paneles_out = {}
for nombre, (X, y, C) in {
    "sin_outlier": (X_base, y_base, C_DURO),
    "duro": (X_out, y_out, C_DURO),
    "suave": (X_out, y_out, 0.5),
}.items():
    modelo, w, b = svm_lineal(X, y, C)
    paneles_out[nombre] = {
        "w": w.tolist(),
        "b": b,
        "sv": redondear(modelo.support_vectors_),
        "n_sv": int(len(modelo.support_)),
        "ancho": float(2 / np.linalg.norm(w)),
        "errores": int(np.sum(y * (X @ w + b) < 0)),
    }
assert paneles_out["duro"]["errores"] == 0, "con margen duro debe separar"

datos_outlier = {
    **por_clase(X_base, y_base),
    "outlier": outlier.tolist(),
    "paneles": paneles_out,
}


# ===========================================================================
# 4. Efecto de C con datos que se solapan
# ===========================================================================
rng = np.random.default_rng(9)
m4 = 30
Xc_neg = rng.normal([1.7, 1.6], 0.85, (m4, 2))
Xc_pos = rng.normal([3.1, 3.0], 0.85, (m4, 2))
X_c = np.vstack([Xc_neg, Xc_pos])
y_c = np.concatenate([-np.ones(m4), np.ones(m4)])

paneles_c = {}
for nombre, C in {"pequeno": 0.05, "grande": 10.0}.items():
    modelo, w, b = svm_lineal(X_c, y_c, C)
    alphas = np.zeros(len(y_c))
    alphas[modelo.support_] = np.abs(modelo.dual_coef_.ravel())  # dual_coef_ = α_i y_i
    paneles_c[nombre] = {
        "C": C,
        "w": w.tolist(),
        "b": b,
        "sv": redondear(modelo.support_vectors_),
        "n_sv": int(len(modelo.support_)),
        "n_acotados": int(np.sum(np.isclose(alphas, C, rtol=1e-3))),
        "ancho": float(2 / np.linalg.norm(w)),
        "exactitud": float(modelo.score(X_c, y_c)),
    }

efecto_C = {**por_clase(X_c, y_c), "paneles": paneles_c, "m": 2 * m4}


def main():
    guardar("separables", separables)
    guardar("gaussiano", gaussiano)
    guardar("outlier", datos_outlier)
    guardar("efecto_C", efecto_C)

    # Resumen en consola
    d = separables["opt"]
    print(f"Margen óptimo: γ = {d['margen']:.3f}, ||w|| = {d['norma_w']:.3f}, "
          f"vectores de soporte = {d['n_sv']}")
    print(f"Otra frontera: γ = {separables['otra']['margen']:.3f}")
    print(f"Punto A: γ_A = {gamma_A:.3f}")
    for k, v in paneles.items():
        print(f"Gaussiano σ={v['sigma']}: SV = {v['n_sv']}, train = {v['exactitud_train']:.2f}, "
              f"test = {v['exactitud_test']:.2f}, segmentos = {len(v['frontera'])}")
    for k, v in paneles_out.items():
        print(f"Outlier {k}: ancho = {v['ancho']:.2f}, SV = {v['n_sv']}, errores = {v['errores']}")
    for k, v in paneles_c.items():
        print(f"C={v['C']}: ancho = {v['ancho']:.2f}, SV = {v['n_sv']} ({v['n_acotados']} con α = C), "
              f"exactitud = {v['exactitud']:.2f}")


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
