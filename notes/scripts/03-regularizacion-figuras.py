"""
Genera los datos de las figuras de la Clase 3 (generalización y regularización).

Cada figura se escribe en notes/figs/<nombre>.json y Typst la lee con json(...).
Se puede ejecutar desde cualquier carpeta:

    python notes/scripts/03-regularizacion-figuras.py

También se puede importar (lo hace notebooks/clase03.py): al importarlo solo se
construyen los datos y las funciones; los JSON se escriben en main().

Convenciones (las mismas de las notas):
    J(θ)      = 1/2 Σ (h(x) - y)^2
    Ridge     = J(θ) + λ/2 Σ_{j>=1} θ_j^2
    Lasso     = J(θ) + λ   Σ_{j>=1} |θ_j|
θ_0 (el intercepto) nunca se penaliza: las features se centran y θ_0 = media de y.
"""
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import Lasso, Ridge

FIGS = Path(__file__).resolve().parent.parent / "figs"

rng = np.random.default_rng(42)
SEMILLA_CASAS = int(os.environ.get("SEMILLA_CASAS", 12))


def guardar(nombre, datos):
    with open(FIGS / f"{nombre}.json", "w") as fh:
        json.dump(datos, fh, indent=1)


def redondear(arr, d=4):
    return np.round(np.asarray(arr, dtype=float), d).tolist()


# ===========================================================================
# 1. Regresión polinomial: ajustes y error vs. grado
# ===========================================================================
def f_real(x):
    return 2.0 + 0.5 * x + 1.0 * np.sin(1.3 * x)


SIGMA = 0.35
m_train, m_val = 15, 300

x_train = np.sort(rng.uniform(0.2, 4.8, m_train))
y_train = f_real(x_train) + rng.normal(0, SIGMA, m_train)
# La validación se toma en el mismo rango que el entrenamiento: así medimos
# interpolación y no extrapolación (fuera del rango los polinomios explotan).
x_val = rng.uniform(x_train.min(), x_train.max(), m_val)
y_val = f_real(x_val) + rng.normal(0, SIGMA, m_val)

# Escalamos x antes de construir las potencias: con x en [0, 5], x^10 ~ 1e7
# y X^T X queda numéricamente singular.
mu_x, sd_x = x_train.mean(), x_train.std()


def potencias(x, grado):
    """Columnas 1, z, z^2, ..., z^grado con z = x estandarizado."""
    z = (x - mu_x) / sd_x
    return np.vander(z, grado + 1, increasing=True)


def ajustar_ols(grado):
    X = potencias(x_train, grado)
    theta, *_ = np.linalg.lstsq(X, y_train, rcond=None)
    return theta


def mse(X, y, theta):
    return float(np.mean((X @ theta - y) ** 2))


x_grid = np.linspace(0, 5, 400)
grados_mostrados = [1, 4, 10]


def curva_error_polinomios():
    curva_error = []
    for g in range(1, 11):
        theta = ajustar_ols(g)
        curva_error.append({
            "grado": g,
            "mse_train": mse(potencias(x_train, g), y_train, theta),
            "mse_val": mse(potencias(x_val, g), y_val, theta),
        })
    return curva_error


# ===========================================================================
# 2. Ridge sobre el polinomio de grado 10
# ===========================================================================
GRADO = 10

# Además de construir las potencias sobre z, centramos y escalamos cada
# columna con las medias y desviaciones de entrenamiento. Con las features
# centradas, el intercepto es la media de y, y la matriz de diseño no lleva
# columna de unos (igual que en las notas).
P_train = potencias(x_train, GRADO)[:, 1:]
col_mu, col_sd = P_train.mean(axis=0), P_train.std(axis=0)
y_media = y_train.mean()


def diseno_ridge(x):
    return (potencias(x, GRADO)[:, 1:] - col_mu) / col_sd


X_tr = diseno_ridge(x_train)
X_va = diseno_ridge(x_val)


def ridge_cerrada(X, y, lam):
    """θ = (X^T X + λ I)^{-1} X^T (y - ȳ), con X centrada y sin columna de unos."""
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ (y - y_media))


def mse_ridge(X, y, theta):
    return float(np.mean((y_media + X @ theta - y) ** 2))


def verificar_ridge():
    """Verificación contra scikit-learn. Para la suma de cuadrados de las notas,
    alpha de sklearn.Ridge coincide con λ."""
    lam = 0.5
    sk = Ridge(alpha=lam).fit(X_tr, y_train)
    assert np.allclose(ridge_cerrada(X_tr, y_train, lam), sk.coef_, atol=1e-6)
    assert np.isclose(sk.intercept_, y_media)


def curva_error_ridge(lambdas=np.logspace(-6, 3, 60)):
    error_lambda = []
    for lam in lambdas:
        th = ridge_cerrada(X_tr, y_train, lam)
        error_lambda.append({
            "lambda": float(lam),
            "mse_train": mse_ridge(X_tr, y_train, th),
            "mse_val": mse_ridge(X_va, y_val, th),
        })
    return error_lambda


# ===========================================================================
# 3. Caminos de coeficientes de Ridge y Lasso (casas con varias features)
# ===========================================================================
m_casas = 80
rng_casas = np.random.default_rng(SEMILLA_CASAS)
nombres = [
    "Área",
    "Habitaciones",
    "Baños",
    "Antigüedad",
    "Distancia al centro",
    "Ruido 1",
    "Ruido 2",
    "Ruido 3",
]
area = rng_casas.normal(0, 1, m_casas)
habitaciones = 0.8 * area + 0.6 * rng_casas.normal(0, 1, m_casas)  # correlacionada con área
banos = 0.5 * habitaciones + 0.85 * rng_casas.normal(0, 1, m_casas)
antiguedad = rng_casas.normal(0, 1, m_casas)
distancia = rng_casas.normal(0, 1, m_casas)
ruido = rng_casas.normal(0, 1, (m_casas, 3))
Xc = np.column_stack([area, habitaciones, banos, antiguedad, distancia, ruido])
theta_real = np.array([3.0, 1.0, 0.8, -1.2, -2.0, 0.0, 0.0, 0.0])
yc = 10.0 + Xc @ theta_real + rng_casas.normal(0, 1.5, m_casas)

# Estandarizamos las features y centramos y: así el intercepto es la media de y
# y no interviene en la penalización.
Xc = (Xc - Xc.mean(axis=0)) / Xc.std(axis=0)
yc_c = yc - yc.mean()

lambdas_camino = np.logspace(-2, 4, 80)


def ridge_casas(lam):
    return np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ yc_c)


def lasso_casas(lam):
    # sklearn.Lasso minimiza 1/(2m) Σ(...)^2 + alpha Σ|θ|, así que alpha = λ/m.
    return Lasso(alpha=lam / m_casas, fit_intercept=False, max_iter=100000,
                 tol=1e-10).fit(Xc, yc_c).coef_


def caminos(lambdas=lambdas_camino):
    """Coeficientes de Ridge y Lasso para cada λ: dos arreglos de len(lambdas) x 8."""
    camino_ridge = np.array([ridge_casas(lam) for lam in lambdas])
    camino_lasso = np.array([lasso_casas(lam) for lam in lambdas])
    return camino_ridge, camino_lasso


# ===========================================================================
# 4. Geometría: curvas de nivel tocando la región L2 (círculo) y L1 (rombo)
# ===========================================================================
theta_ols = np.array([1.9, 1.8])
ang = np.deg2rad(75)
R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
A = R @ np.diag([1.0, 3.0]) @ R.T  # J(θ) = (θ - θ_ols)^T A (θ - θ_ols)
T = 1.0  # tamaño de la región permitida


def J_cuad(th):
    d = th - theta_ols
    return float(d @ A @ d)


def minimo_en_frontera(frontera, radio=T):
    ts = np.linspace(0, 2 * np.pi, 20000, endpoint=False)
    puntos = frontera(ts, radio).T
    valores = np.einsum("ij,jk,ik->i", puntos - theta_ols, A, puntos - theta_ols)
    return puntos[np.argmin(valores)]


def circulo(s, radio=T):
    return radio * np.array([np.cos(s), np.sin(s)])


def rombo(s, radio=T):
    c, d = np.cos(s), np.sin(s)
    return radio * np.array([c, d]) / (abs(c) + abs(d))


def elipse(nivel, n=200):
    """Curva {θ : J(θ) = nivel}."""
    w, V = np.linalg.eigh(A)
    s = np.linspace(0, 2 * np.pi, n)
    u = np.stack([np.sqrt(nivel / w[0]) * np.cos(s), np.sqrt(nivel / w[1]) * np.sin(s)])
    return redondear((theta_ols[:, None] + V @ u).T)


# ===========================================================================
# Escritura de los JSON
# ===========================================================================
def main():
    # 1. Regresión polinomial
    ajustes = {}
    for g in grados_mostrados:
        theta = ajustar_ols(g)
        ajustes[str(g)] = {
            "curva": redondear(np.column_stack([x_grid, potencias(x_grid, g) @ theta])),
            "max_abs_theta": float(np.abs(theta).max()),
        }
    curva_error = curva_error_polinomios()
    guardar("polinomios", {
        "train": redondear(np.column_stack([x_train, y_train])),
        "real": redondear(np.column_stack([x_grid, f_real(x_grid)])),
        "ajustes": ajustes,
        "curva_error": curva_error,
        "ruido_var": SIGMA**2,
    })

    # 2. Ridge sobre el polinomio de grado 10
    verificar_ridge()
    error_lambda = curva_error_ridge()
    mejor = min(error_lambda, key=lambda r: r["mse_val"])
    lambdas_mostrados = {"pequeno": 1e-6, "mejor": mejor["lambda"], "grande": 10.0}
    ajustes_ridge = {}
    for nombre, lam in lambdas_mostrados.items():
        th = ridge_cerrada(X_tr, y_train, lam)
        ajustes_ridge[nombre] = {
            "lambda": lam,
            "curva": redondear(np.column_stack([x_grid, y_media + diseno_ridge(x_grid) @ th])),
            "max_abs_theta": float(np.abs(th).max()),
        }
    guardar("ridge_polinomio", {
        "train": redondear(np.column_stack([x_train, y_train])),
        "real": redondear(np.column_stack([x_grid, f_real(x_grid)])),
        "ajustes": ajustes_ridge,
        "curva_error": error_lambda,
        "mejor_lambda": mejor["lambda"],
        "mejor_mse_val": mejor["mse_val"],
        "ruido_var": SIGMA**2,
    })

    # 3. Caminos de coeficientes
    camino_ridge, camino_lasso = caminos()
    # λ a partir del cual cada coeficiente de Lasso es exactamente cero
    lambda_cero = []
    for j in range(len(nombres)):
        no_cero = np.nonzero(np.abs(camino_lasso[:, j]) > 1e-10)[0]
        lambda_cero.append(float(lambdas_camino[no_cero[-1] + 1]) if len(no_cero) and no_cero[-1] + 1 < len(lambdas_camino) else None)
    guardar("caminos", {
        "nombres": nombres,
        "theta_real": theta_real.tolist(),
        "lambdas": lambdas_camino.tolist(),
        "ridge": [redondear(np.column_stack([lambdas_camino, camino_ridge[:, j]])) for j in range(len(nombres))],
        "lasso": [redondear(np.column_stack([lambdas_camino, camino_lasso[:, j]])) for j in range(len(nombres))],
        "lambda_cero_lasso": lambda_cero,
        "m": m_casas,
    })

    # 4. Geometría
    geometria = {"theta_ols": theta_ols.tolist(), "t": T}
    for nombre, frontera in [("l2", circulo), ("l1", rombo)]:
        sol = minimo_en_frontera(frontera)
        nivel = J_cuad(sol)
        geometria[nombre] = {
            "solucion": redondear(sol, 3),
            "contornos": [elipse(nivel * f) for f in (0.12, 0.4, 1.0, 1.8)],
        }
    guardar("geometria", geometria)

    # 5. Interpretación probabilística: puntos alrededor de una recta
    th0, th1, sig = 1.0, 0.6, 0.4
    xs = rng.uniform(0.3, 4.7, 40)
    ys = th0 + th1 * xs + rng.normal(0, sig, len(xs))
    guardar("gaussiana", {
        "puntos": redondear(np.column_stack([xs, ys])),
        "theta0": th0,
        "theta1": th1,
        "sigma": sig,
    })

    # Resumen en consola
    print("Polinomios (MSE validación):",
          {r["grado"]: round(r["mse_val"], 3) for r in curva_error})
    print(f"Ridge: mejor λ = {mejor['lambda']:.3g}, MSE val = {mejor['mse_val']:.3f}")
    for k, v in ajustes_ridge.items():
        print(f"  {k:8s} λ = {v['lambda']:.3g}  max|θ| = {v['max_abs_theta']:.3g}")
    print("Lasso: λ desde el cual el coeficiente es cero")
    for n, l in zip(nombres, lambda_cero):
        print(f"  {n:22s} {l}")
    print("Geometría: solución L2", geometria["l2"]["solucion"], " L1", geometria["l1"]["solucion"])


if __name__ == "__main__":
    main()
