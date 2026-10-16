"""
Genera los datos de las figuras de las notas de clasificación lineal y
regresión logística.

Cada figura se escribe en figs/<nombre>.json y Typst la lee con json(...).
Ejecutar desde la carpeta de las notas:

    python generar_figuras.py

Convenciones (las mismas de las notas):
    h(x)  = g(θ^T x),  g(z) = 1 / (1 + e^{-z}),  x_0 = 1
    ℓ(θ)  = Σ [ y log h(x) + (1 - y) log(1 - h(x)) ]        (log-verosimilitud)
    Se maximiza con ascenso por gradiente: θ := θ + α ∇ℓ(θ).
    Con regularización L2 se maximiza ℓ(θ) - λ/2 Σ_{j>=1} θ_j^2.
    En scikit-learn, LogisticRegression(C=...) minimiza -C ℓ(θ) + 1/2 ||w||^2,
    así que C = 1/λ.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression


def guardar(nombre, datos):
    with open(f"figs/{nombre}.json", "w") as fh:
        json.dump(datos, fh, indent=1)


def redondear(arr, d=4):
    return np.round(np.asarray(arr, dtype=float), d).tolist()


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))


def log_verosimilitud(theta, X, y, lam=0.0):
    """ℓ(θ) de las notas (menos la penalización L2 sin θ_0)."""
    z = X @ theta
    # log h = -log(1 + e^{-z}) y log(1 - h) = -log(1 + e^{z}), de forma estable
    ell = -(y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z))
    return float(ell.sum() - lam / 2 * np.sum(theta[1:] ** 2))


def gradiente(theta, X, y, lam=0.0):
    """∇ℓ(θ) = Σ (y - h) x  (menos λθ_j para j >= 1)."""
    g = X.T @ (y - sigmoide(X @ theta))
    g[1:] -= lam * theta[1:]
    return g


def con_unos(X):
    return np.column_stack([np.ones(len(X)), X])


# ===========================================================================
# 1. Regresión lineal aplicada a etiquetas 0/1 (y por qué falla)
# ===========================================================================
x_neg = np.array([0.8, 1.4, 2.0, 2.5, 3.0, 3.5, 4.0, 4.4])
x_pos = np.array([4.9, 5.3, 5.8, 6.3, 6.8, 7.3, 7.8, 8.3])
x_lejos = np.array([24.0, 26.5, 29.0])

x_a = np.concatenate([x_neg, x_pos])
y_a = np.concatenate([np.zeros(8), np.ones(8)])
x_b = np.concatenate([x_a, x_lejos])
y_b = np.concatenate([y_a, np.ones(3)])


def recta_mc(x, y):
    """Mínimos cuadrados con una feature: devuelve (θ_0, θ_1)."""
    X = con_unos(x)
    return np.linalg.lstsq(X, y, rcond=None)[0]


regresion = {}
for nombre, (x, y) in {"a": (x_a, y_a), "b": (x_b, y_b)}.items():
    t0, t1 = recta_mc(x, y)
    umbral = (0.5 - t0) / t1  # punto donde la recta vale 0.5
    pred = (t0 + t1 * x >= 0.5).astype(int)
    regresion[nombre] = {
        "puntos": redondear(np.column_stack([x, y])),
        "theta": [float(t0), float(t1)],
        "umbral": float(umbral),
        "errores": int((pred != y).sum()),
    }
guardar("regresion_falla", regresion)


# ===========================================================================
# 2. Frontera lineal en 2D (datos separables)
# ===========================================================================
rng = np.random.default_rng(11)
m2 = 20
X0 = rng.normal([1.4, 1.3], 0.45, (m2, 2))
X1 = rng.normal([3.4, 3.1], 0.45, (m2, 2))
X_sep = np.vstack([X0, X1])
y_sep = np.concatenate([np.zeros(m2), np.ones(m2)])

# Frontera de referencia, escogida a mano: θ_0 + θ_1 x_1 + θ_2 x_2 = 0
theta_ref = np.array([-4.6, 1.0, 1.0])
assert np.all((con_unos(X_sep) @ theta_ref >= 0) == (y_sep == 1)), "theta_ref no separa"

# Otras fronteras que también separan los datos
otras = [np.array([-5.2, 1.6, 0.6]), np.array([-3.9, 0.45, 1.3]), np.array([-4.3, 1.0, 0.85])]
for t in otras:
    assert np.all((con_unos(X_sep) @ t >= 0) == (y_sep == 1)), f"{t} no separa"

guardar("frontera", {
    "clase0": redondear(X0),
    "clase1": redondear(X1),
    "theta": theta_ref.tolist(),
    "otras": [t.tolist() for t in otras],
})


# ===========================================================================
# 3. Regresión logística sobre datos que se solapan
# ===========================================================================
rng = np.random.default_rng(5)
m3 = 40
Z0 = rng.normal([1.8, 1.7], 0.8, (m3, 2))
Z1 = rng.normal([3.0, 2.9], 0.8, (m3, 2))
X_sol = np.vstack([Z0, Z1])
y_sol = np.concatenate([np.zeros(m3), np.ones(m3)])

# Sin regularización (C muy grande) para que coincida con el MLE de las notas
modelo = LogisticRegression(C=1e8, max_iter=10000, tol=1e-12).fit(X_sol, y_sol)
theta_log = np.concatenate([modelo.intercept_, modelo.coef_.ravel()])

# Verificación: el gradiente de las notas es ~0 en el óptimo de sklearn
assert np.linalg.norm(gradiente(theta_log, con_unos(X_sol), y_sol)) < 1e-3

h_sol = sigmoide(con_unos(X_sol) @ theta_log)
guardar("logistica", {
    "clase0": redondear(Z0),
    "clase1": redondear(Z1),
    "theta": theta_log.tolist(),
    "ell": log_verosimilitud(theta_log, con_unos(X_sol), y_sol),
    "exactitud": float(np.mean((h_sol >= 0.5) == (y_sol == 1))),
    "m": 2 * m3,
})


# ===========================================================================
# 4. Ascenso por gradiente: datos que se solapan vs. datos separables
# ===========================================================================
def ascenso(X, y, alpha, pasos, lam=0.0):
    theta = np.zeros(X.shape[1])
    historia = []
    for k in range(pasos + 1):
        historia.append((k, log_verosimilitud(theta, X, y, lam), float(np.linalg.norm(theta[1:]))))
        theta = theta + alpha * gradiente(theta, X, y, lam)
    return theta, historia


PASOS = 20000
# Centramos las features para que el descenso converja razonablemente rápido.
Xs = con_unos(X_sol - X_sol.mean(axis=0))
Xp = con_unos(X_sep - X_sep.mean(axis=0))
_, h_solapan = ascenso(Xs, y_sol, 0.01, PASOS)
_, h_separa = ascenso(Xp, y_sep, 0.01, PASOS)
_, h_separa_reg = ascenso(Xp, y_sep, 0.01, PASOS, lam=1.0)

submuestra = sorted(set(np.unique(np.geomspace(1, PASOS, 60).astype(int)).tolist() + [0]))


def tomar(hist, columna):
    return [[hist[k][0], hist[k][columna]] for k in submuestra if k > 0]


guardar("ascenso", {
    "ell_solapan": tomar(h_solapan, 1),
    "ell_separa": tomar(h_separa, 1),
    "norma_solapan": tomar(h_solapan, 2),
    "norma_separa": tomar(h_separa, 2),
    "norma_separa_reg": tomar(h_separa_reg, 2),
    "pasos": PASOS,
})


# ===========================================================================
# 5. Datos separables con una parábola y feature map φ(x) = (1, x_1^2, x_2)
# ===========================================================================
rng = np.random.default_rng(8)
m5 = 140
Xc = np.column_stack([rng.uniform(-2, 2, m5), rng.uniform(-1, 3, m5)])
# Etiqueta 1 por encima de la parábola x_2 = x_1^2 - 0.5, con algo de ruido
yc = (Xc[:, 1] - (Xc[:, 0] ** 2 - 0.5) + rng.normal(0, 0.25, m5) > 0).astype(float)

# Modelo lineal en x
lin = LogisticRegression(C=1e8, max_iter=10000).fit(Xc, yc)
theta_lin = np.concatenate([lin.intercept_, lin.coef_.ravel()])

# Modelo lineal en φ(x) = (x_1^2, x_2)
Phi = np.column_stack([Xc[:, 0] ** 2, Xc[:, 1]])
cuad = LogisticRegression(C=1e8, max_iter=10000).fit(Phi, yc)
theta_cuad = np.concatenate([cuad.intercept_, cuad.coef_.ravel()])

# La frontera θ_0 + θ_1 x_1^2 + θ_2 x_2 = 0 es la parábola
# x_2 = -(θ_0 + θ_1 x_1^2) / θ_2
t0, t1, t2 = theta_cuad
x1 = np.linspace(-2, 2, 200)
parabola = np.column_stack([x1, -(t0 + t1 * x1 ** 2) / t2])

guardar("parabola", {
    "clase0": redondear(Xc[yc == 0]),
    "clase1": redondear(Xc[yc == 1]),
    "phi0": redondear(Phi[yc == 0]),
    "phi1": redondear(Phi[yc == 1]),
    "theta_lin": theta_lin.tolist(),
    "theta_cuad": theta_cuad.tolist(),
    "parabola": redondear(parabola),
    "exactitud_lin": float(lin.score(Xc, yc)),
    "exactitud_cuad": float(cuad.score(Phi, yc)),
})


# ===========================================================================
# Resumen en consola
# ===========================================================================
for k, v in regresion.items():
    print(f"Regresión ({k}): umbral = {v['umbral']:.2f}, errores = {v['errores']}")
print("Logística: θ =", np.round(theta_log, 3), " exactitud =", round(float(np.mean((h_sol >= 0.5) == (y_sol == 1))), 3))
print(f"Ascenso: ℓ final solapan = {h_solapan[-1][1]:.3f}, separables = {h_separa[-1][1]:.4f}")
print(f"          ||θ|| final solapan = {h_solapan[-1][2]:.2f}, separables = {h_separa[-1][2]:.2f}, "
      f"separables con λ=1: {h_separa_reg[-1][2]:.2f}")
print(f"Parábola: exactitud lineal = {lin.score(Xc, yc):.3f}, con φ = {cuad.score(Phi, yc):.3f}, "
      f"θ = {np.round(theta_cuad, 2)}")
