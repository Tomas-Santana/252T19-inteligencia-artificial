"""
Experimentos: ¿qué le pasa a θ cuando los datos son separables?
Clase 4: regresión logística.

Uso:

    python 04-experimentos-escala.py

El script genera dos nubes de puntos en el plano (clase 0 y clase 1), entrena
regresión logística con ascenso por gradiente y corre cuatro experimentos, uno
por figura. Cierra cada figura para pasar a la siguiente.

Para experimentar, cambia los valores de la sección PARÁMETROS y vuelve a
correr el script. No hace falta tocar nada más.

Recordatorio de las notas:
    h(x) = g(θ^T x),  g(z) = 1 / (1 + e^{-z})
    ℓ(θ) = Σ y log h(x) + (1 - y) log(1 - h(x))
    Con regularización L2 se maximiza ℓ(θ) - λ/2 (θ_1^2 + θ_2^2).
    Escribimos θ = c (b, cos φ, sin φ): la dirección φ y el desplazamiento b fijan
    la frontera, y c = ||(θ_1, θ_2)|| es la escala. Cambiar c no mueve la frontera.

Preguntas para explorar:

 1. Con SEPARACION = 5 los datos son separables. En el experimento 1, ¿ℓ(c)
    tiene un máximo? ¿Y con SEPARACION = 1.5?
 2. En el experimento 2, con λ = 0 y datos separables, ¿la frontera sigue
    moviéndose después de unas 100 iteraciones? Si no se mueve, ¿qué es lo que
    sigue cambiando?
 3. Sube PASOS a 200000. ¿||θ|| se estabiliza? ¿Crece rápido o lento?
 4. Con datos separables, pon AGREGAR_ATIPICO = True: se agrega un solo punto
    con la etiqueta equivocada. ¿Qué pasa con ||θ|| cuando λ = 0? ¿Por qué un
    solo punto cambia tanto el resultado?
 5. En el experimento 3, ¿dónde queda el máximo de ℓ(c) - λ/2 c^2 para cada λ?
    ¿Qué pasa con ese máximo cuando λ crece?
 6. En el experimento 4 (scikit-learn), ¿qué valor de C corresponde a λ = 1?
    ¿Qué valor de C usa scikit-learn si no le pasas ninguno? Con C muy grande y
    datos separables, ¿qué probabilidad le asigna el modelo al punto más cercano
    a la frontera?
 7. ¿Cambia la exactitud de entrenamiento al subir λ? ¿Y las probabilidades?
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

# ===========================================================================
# PARÁMETROS (cámbialos y vuelve a correr)
# ===========================================================================
SEPARACION = 5.0          # distancia entre los centros de las clases: grande = separables, pequeña = se solapan
M_POR_CLASE = 20          # ejemplos por clase
SEMILLA = 0               # cambia la muestra aleatoria
AGREGAR_ATIPICO = False   # agrega un punto de la clase 1 metido entre los de la clase 0
LAMBDAS = [0.0, 0.1, 1.0] # valores de λ que se comparan en los experimentos 2 y 3
ALPHA = 0.01              # tasa de aprendizaje
PASOS = 20000             # iteraciones del ascenso por gradiente

C0, C1 = "#1f77b4", "#d62728"
COLORES_LAMBDA = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]


# ===========================================================================
# Datos y funciones básicas
# ===========================================================================
def generar_datos():
    rng = np.random.default_rng(SEMILLA)
    u = np.array([1.0, 1.0]) / np.sqrt(2)  # las clases se separan en la dirección diagonal
    X0 = rng.normal(0, 1, (M_POR_CLASE, 2)) - SEPARACION / 2 * u
    X1 = rng.normal(0, 1, (M_POR_CLASE, 2)) + SEPARACION / 2 * u
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(M_POR_CLASE), np.ones(M_POR_CLASE)])
    if AGREGAR_ATIPICO:
        X = np.vstack([X, X0.mean(axis=0)])  # un punto de la clase 1 justo en el centro de la clase 0
        y = np.append(y, 1.0)
    return X, y


def con_unos(X):
    return np.column_stack([np.ones(len(X)), X])


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def log_verosimilitud(theta, X1, y):
    """ℓ(θ), calculada de forma estable. X1 ya incluye la columna de unos."""
    z = X1 @ theta
    return float(-(y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z)).sum())


def ascenso(X1, y, lam):
    """Ascenso por gradiente desde θ = 0. Devuelve el θ final y la historia de ||θ|| y ℓ."""
    theta = np.zeros(X1.shape[1])
    normas, ells = [], []
    for _ in range(PASOS):
        gradiente = X1.T @ (y - sigmoide(X1 @ theta))
        gradiente[1:] -= lam * theta[1:]  # el intercepto no se penaliza
        theta = theta + ALPHA * gradiente
        normas.append(np.linalg.norm(theta[1:]))
        ells.append(log_verosimilitud(theta, X1, y))
    return theta, np.array(normas), np.array(ells)


def separar_escala(theta):
    """Escribe θ = c (b, cos φ, sin φ). Devuelve c y la dirección unitaria (b, cos φ, sin φ)."""
    c = np.linalg.norm(theta[1:])
    return c, theta / c


def dibujar_puntos(ax, X, y):
    ax.scatter(*X[y == 0].T, s=90, marker="o", color=C0, ec="white", zorder=3, label="$y = 0$")
    ax.scatter(*X[y == 1].T, s=100, marker="^", color=C1, ec="white", zorder=3, label="$y = 1$")


def dibujar_recta(ax, theta, nivel=0.0, **estilo):
    """Recta θ_0 + θ_1 x_1 + θ_2 x_2 = nivel."""
    x1 = np.array(ax.get_xlim())
    ax.plot(x1, (nivel - theta[0] - theta[1] * x1) / theta[2], **estilo)


# ===========================================================================
# Experimento 1: ℓ como función de la escala c, con la frontera fija
# ===========================================================================
def experimento_1(X, y):
    """Tomamos una frontera razonable (la de λ = 1) y solo cambiamos c."""
    X1 = con_unos(X)
    theta_ref, _, _ = ascenso(X1, y, lam=1.0)
    _, direccion = separar_escala(theta_ref)
    errores = int(np.sum((X1 @ direccion >= 0) != (y == 1)))

    cs = np.linspace(0, 30, 400)
    ells = [log_verosimilitud(c * direccion, X1, y) for c in cs]
    k = int(np.argmax(ells))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cs, ells, lw=2.5, color="black")
    ax.axhline(0, color="0.7", ls=":")
    if k < len(cs) - 1:
        ax.plot(cs[k], ells[k], "o", color=C1, ms=10, label=f"máximo en c ≈ {cs[k]:.2f}")
        ax.legend()
        print(f"Experimento 1: la frontera comete {errores} errores y ℓ(c) tiene un máximo en c ≈ {cs[k]:.2f}.")
    else:
        print(f"Experimento 1: la frontera comete {errores} errores y ℓ(c) sigue subiendo en todo el rango.")
    ax.set(xlabel="escala c", ylabel=r"$\ell(c \cdot \theta_{\mathrm{dirección}})$",
           title=f"Experimento 1: misma frontera, distinta escala ({errores} errores de clasificación)")
    plt.tight_layout()
    plt.show()


# ===========================================================================
# Experimento 2: ascenso por gradiente con distintos λ
# ===========================================================================
def experimento_2(X, y):
    X1 = con_unos(X)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    lim = np.abs(X).max() + 0.5
    ax1.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xlabel="$x_1$", ylabel="$x_2$",
            title="Fronteras finales (punteadas: h = 0.1 y h = 0.9)")
    dibujar_puntos(ax1, X, y)
    if AGREGAR_ATIPICO:
        ax1.scatter(*X[-1], s=350, facecolor="none", ec="black", lw=2, zorder=4)

    print("Experimento 2:")
    for lam, color in zip(LAMBDAS, COLORES_LAMBDA):
        theta, normas, ells = ascenso(X1, y, lam)
        exactitud = np.mean((X1 @ theta >= 0) == (y == 1))
        print(f"  λ = {lam:<5g}  ||θ|| final = {normas[-1]:7.3f}   ℓ final = {ells[-1]:9.4f}   exactitud = {exactitud:.0%}")
        dibujar_recta(ax1, theta, color=color, lw=2, label=f"λ = {lam:g}")
        for nivel in (-np.log(9), np.log(9)):
            dibujar_recta(ax1, theta, nivel, color=color, lw=1, ls="--")
        ax2.semilogx(np.arange(1, PASOS + 1), normas, color=color, lw=2.5, label=f"λ = {lam:g}")
    ax1.set(xlim=(-lim, lim), ylim=(-lim, lim))
    ax1.legend(loc="best", fontsize=9)
    ax2.set(xlabel="iteración", ylabel=r"$\|\theta\| = c$", title="Escala de θ durante el entrenamiento")
    ax2.legend()
    fig.suptitle("Experimento 2: ascenso por gradiente")
    plt.tight_layout()
    plt.show()


# ===========================================================================
# Experimento 3: la penalización contra la ganancia en ℓ
# ===========================================================================
def experimento_3(X, y):
    """Con la frontera fija, maximizamos ℓ(c) - λ/2 c^2 para varios λ."""
    X1 = con_unos(X)
    theta_ref, _, _ = ascenso(X1, y, lam=1.0)
    _, direccion = separar_escala(theta_ref)
    cs = np.linspace(0, 30, 400)
    ells = np.array([log_verosimilitud(c * direccion, X1, y) for c in cs])

    fig, ax = plt.subplots(figsize=(8, 5))
    print("Experimento 3:")
    for lam, color in zip(LAMBDAS, COLORES_LAMBDA):
        objetivo = ells - lam / 2 * cs**2
        k = int(np.argmax(objetivo))
        ax.plot(cs, objetivo, color=color, lw=2.5, label=f"λ = {lam:g}")
        ax.plot(cs[k], objetivo[k], "o", color=color, ms=9)
        al_borde = " (en el borde del rango: sigue subiendo)" if k == len(cs) - 1 else ""
        print(f"  λ = {lam:<5g}  máximo en c ≈ {cs[k]:.2f}{al_borde}")
    ax.set(xlabel="escala c", ylabel=r"$\ell - \frac{\lambda}{2} c^2$",
           ylim=(min(ells.min(), -5) * 1.1, 1), title="Experimento 3: ganancia en ℓ contra penalización")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ===========================================================================
# Experimento 4: lo mismo con scikit-learn (C = 1/λ)
# ===========================================================================
def experimento_4(X, y):
    print("Experimento 4 (scikit-learn, C = 1/λ):")
    print("  C          ||w||      h(x) del punto más cercano a la frontera")
    for C in [0.01, 0.1, 1.0, 10.0, 1e3, 1e6]:
        modelo = LogisticRegression(C=C, max_iter=100000).fit(X, y)
        w, b = modelo.coef_.ravel(), modelo.intercept_[0]
        distancias = (X @ w + b) / np.linalg.norm(w)
        i = int(np.argmin(np.abs(distancias)))
        h = modelo.predict_proba(X[i:i + 1])[0, 1]
        print(f"  {C:<10g} {np.linalg.norm(w):8.3f}   h = {h:.6f}  (etiqueta {int(y[i])}, distancia {distancias[i]:+.2f})")


if __name__ == "__main__":
    X, y = generar_datos()
    experimento_1(X, y)
    experimento_2(X, y)
    experimento_3(X, y)
    experimento_4(X, y)
