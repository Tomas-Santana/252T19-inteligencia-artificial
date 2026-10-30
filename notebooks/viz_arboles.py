"""Visualizaciones interactivas de la clase 6: árboles de decisión y ensamblaje.

Uso, desde 06-visualizaciones.ipynb:

    import viz_arboles as viz
    viz.explorar_division()

Cada función `xxx()` muestra controles (ipywidgets) y una figura. La función
`dibujar_xxx(...)` correspondiente hace la figura con parámetros fijos, por si se
quiere usar sin widgets (por ejemplo, para guardar una imagen).

Requiere numpy, matplotlib, scikit-learn, ipywidgets y xgboost.
"""

from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

C_NEG = "#1f77b4"
C_POS = "#d62728"
C_3 = "#2a9d3a"
FONDO = ListedColormap(["#d6e4f2", "#f6d5d5"])

plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False})


# ---------------------------------------------------------------------------
# Datos (los mismos de las notas)
# ---------------------------------------------------------------------------


def datos_esqui(m=140, semilla=3):
    """Mes (1 a 13) y latitud (-90 a 90). y = 1 si se puede esquiar."""
    rng = np.random.default_rng(semilla)
    mes = rng.uniform(1, 13, m)
    lat = rng.uniform(-90, 90, m)
    norte = (lat > 30) & ((mes < 3) | (mes >= 12))
    sur = (lat < -30) & (mes >= 6) & (mes < 9)
    return np.c_[mes, lat], (norte | sur).astype(int)


def datos_nieve(m=80, semilla=7):
    """Centímetros de nieve por día según el mes, en el hemisferio norte."""
    rng = np.random.default_rng(semilla)
    x = np.sort(rng.uniform(1, 13, m))
    verdad = 20 * np.maximum(0, np.cos(2 * np.pi * (x - 1) / 12)) ** 1.5
    y = np.maximum(0, verdad + rng.normal(0, 2.5, m))
    return x.reshape(-1, 1), y


def datos_lunas(m=200, semilla=2, ruido=0.35):
    return make_moons(n_samples=m, noise=ruido, random_state=semilla)


# ---------------------------------------------------------------------------
# Utilidades de dibujo
# ---------------------------------------------------------------------------


def _puntos(ax, X, y, tam=22, pesos=None):
    for clase, color, marca in ((0, C_NEG, "o"), (1, C_POS, "^")):
        sel = y == clase
        s = tam if pesos is None else tam * np.asarray(pesos)[sel]
        ax.scatter(X[sel, 0], X[sel, 1], c=color, marker=marca, s=s, edgecolors="none", zorder=3)


def _malla(limites, n=250):
    x0, x1, y0, y1 = limites
    xs, ys = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    return xs, ys, np.c_[xs.ravel(), ys.ravel()]


def _fondo(ax, f, limites, n=250, contorno=True):
    """Pinta la clase predicha por f (que devuelve 0/1 o una probabilidad)."""
    xs, ys, Z = _malla(limites, n)
    v = f(Z).reshape(xs.shape)
    ax.contourf(xs, ys, v > 0.5, levels=[-0.5, 0.5, 1.5], cmap=FONDO, zorder=0)
    if contorno:
        ax.contour(xs, ys, v, levels=[0.5], colors="k", linewidths=1.2, zorder=2)
    ax.set_xlim(limites[0], limites[1])
    ax.set_ylim(limites[2], limites[3])


def _ejes_esqui(ax):
    ax.set_xlim(1, 13)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("mes")
    ax.set_ylabel("latitud")
    ax.set_yticks(range(-90, 91, 30))


def _widgets():
    try:
        import ipywidgets as w
    except ImportError as e:  # pragma: no cover
        raise ImportError("Hace falta ipywidgets: pip install ipywidgets") from e
    return w


def _slider_int(w, desc, a, b, v, paso=1):
    return w.IntSlider(value=v, min=a, max=b, step=paso, description=desc, continuous_update=False,
                       style={"description_width": "initial"})


def _slider(w, desc, a, b, v, paso):
    return w.FloatSlider(value=v, min=a, max=b, step=paso, description=desc, continuous_update=False,
                         style={"description_width": "initial"}, readout_format=".2f")


# ---------------------------------------------------------------------------
# Impurezas
# ---------------------------------------------------------------------------


def gini(p):
    p = np.asarray(p, dtype=float)
    return 2 * p * (1 - p)


def entropia(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def error_clasif(p):
    p = np.asarray(p, dtype=float)
    return np.minimum(p, 1 - p)


IMPUREZAS = {"Gini": gini, "Entropía": entropia, "Error de clasificación": error_clasif}


def _ganancia(y, izq, L):
    n = len(y)
    if izq.all() or (~izq).all():
        return 0.0
    h = lambda yy: L(yy.mean())  # noqa: E731
    return float(h(y) - (izq.sum() / n * h(y[izq]) + (~izq).sum() / n * h(y[~izq])))


# ---------------------------------------------------------------------------
# V1. Tú eliges la pregunta (raíz del árbol del esquí)
# ---------------------------------------------------------------------------


def dibujar_division(feature="latitud", umbral=30.0):
    X, y = datos_esqui()
    j = 0 if feature == "mes" else 1
    izq = X[:, j] < umbral

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})

    # Panel izquierdo: la división y los conteos de cada lado
    for lado, sel in (("izq", izq), ("der", ~izq)):
        if sel.sum() == 0:
            continue
        mayoria = int(y[sel].mean() > 0.5)
        if j == 0:
            x0, x1 = (1, umbral) if lado == "izq" else (umbral, 13)
            ax.axvspan(x0, x1, color=FONDO(mayoria), zorder=0)
            pos = ((x0 + x1) / 2, 80)
        else:
            y0, y1 = (-90, umbral) if lado == "izq" else (umbral, 90)
            ax.axhspan(y0, y1, color=FONDO(mayoria), zorder=0)
            pos = (7, (y0 + y1) / 2)
        texto = f"{int(y[sel].sum())}+ / {int((1 - y[sel]).sum())}−"
        ax.text(*pos, texto, ha="center", va="center", fontsize=11, zorder=5,
                bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9))
    if j == 0:
        ax.axvline(umbral, color="k", lw=1.5)
    else:
        ax.axhline(umbral, color="k", lw=1.5)
    _puntos(ax, X, y)
    _ejes_esqui(ax)
    ax.set_title(f"Pregunta: ¿{feature} < {umbral:g}?   (padre: {y.sum()}+ / {len(y) - y.sum()}−)", fontsize=10)

    # Panel derecho: ganancia de cada medida para todos los umbrales de esa feature
    rango = np.linspace(1, 13, 241) if j == 0 else np.linspace(-90, 90, 361)
    for (nombre, L), color, estilo in zip(IMPUREZAS.items(), (C_POS, C_NEG, "gray"), ("-", "-", "--")):
        g = [_ganancia(y, X[:, j] < t, L) for t in rango]
        actual = _ganancia(y, izq, L)
        ax2.plot(rango, g, color=color, ls=estilo, lw=1.8, label=f"{nombre}: {actual:.4f}")
    ax2.axvline(umbral, color="k", lw=1, ls=":")
    ax2.set_xlabel(f"umbral para {feature}")
    ax2.set_ylabel("ganancia")
    ax2.set_title("Ganancia de cada umbral (en la leyenda, la del umbral actual)", fontsize=10)
    ax2.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    plt.show()


def _colorear_arbol(cajas):
    """plot_tree usa naranja/azul; los cambiamos por los colores de las notas (azul = −, rojo = +)."""
    for c in cajas:
        texto = c.get_text()
        caja = c.get_bbox_patch()
        if caja is None or "class" not in texto:
            continue
        caja.set_facecolor(FONDO(1) if "class = +" in texto else FONDO(0))
        caja.set_edgecolor(C_POS if "class = +" in texto else C_NEG)


def explorar_division():
    """V1. Escoge la primera pregunta del árbol del esquí y mira su ganancia."""
    w = _widgets()
    feature = w.ToggleButtons(options=["latitud", "mes"], description="feature")
    umbral = _slider(w, "umbral", -90, 90, 30, 1)

    def cambiar(cambio):
        if cambio["new"] == "mes":
            umbral.min, umbral.max, umbral.step, umbral.value = 1, 13, 0.25, 3
        else:
            umbral.max, umbral.min, umbral.step, umbral.value = 90, -90, 1, 30

    feature.observe(cambiar, names="value")
    salida = w.interactive_output(dibujar_division, {"feature": feature, "umbral": umbral})
    from IPython.display import display

    display(w.HBox([feature, umbral]), salida)


# ---------------------------------------------------------------------------
# V1b. El árbol completo, nivel por nivel
# ---------------------------------------------------------------------------


def dibujar_arbol_esqui(profundidad=2, criterio="gini"):
    X, y = datos_esqui()
    arbol = DecisionTreeClassifier(max_depth=profundidad, criterion=criterio, random_state=0).fit(X, y)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.5]})
    _fondo(ax, lambda Z: arbol.predict(Z), (1, 13, -90, 90), n=400, contorno=False)
    _puntos(ax, X, y)
    _ejes_esqui(ax)
    ax.set_title(f"{arbol.get_n_leaves()} hojas · exactitud de entrenamiento {arbol.score(X, y):.0%}", fontsize=10)
    cajas = plot_tree(arbol, feature_names=["mes", "latitud"], class_names=["−", "+"], filled=True,
                      impurity=True, proportion=False, rounded=True, ax=ax2, fontsize=7, max_depth=4)
    _colorear_arbol(cajas)
    ax2.set_title("Árbol (se muestran hasta 4 niveles)", fontsize=10)
    fig.tight_layout()
    plt.show()


def arbol_por_niveles():
    """V1b. Árbol de sklearn sobre el esquí, con profundidad máxima ajustable."""
    w = _widgets()
    w.interact(
        dibujar_arbol_esqui,
        profundidad=_slider_int(w, "profundidad máxima", 1, 8, 2),
        criterio=w.ToggleButtons(options=["gini", "entropy"], description="criterio"),
    )


# ---------------------------------------------------------------------------
# V2. Concavidad
# ---------------------------------------------------------------------------


def dibujar_concavidad(p1=0.6, p2=1.0, fraccion_r1=0.5, impureza="Gini"):
    L = IMPUREZAS[impureza]
    pp = fraccion_r1 * p1 + (1 - fraccion_r1) * p2
    hijos = fraccion_r1 * L(p1) + (1 - fraccion_r1) * L(p2)
    g = abs(L(pp) - hijos) if abs(L(pp) - hijos) < 1e-12 else L(pp) - hijos

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ps = np.linspace(0, 1, 400)
    ax.plot(ps, L(ps), "k", lw=2)
    ax.plot([p1, p2], [L(p1), L(p2)], color=C_NEG, ls="--", lw=1.5)
    ax.vlines(pp, 0, hijos, color="gray", ls=":", lw=1)
    ax.vlines(pp, hijos, L(pp), color=C_POS, lw=3, label=f"ganancia = {g:.4f}")
    ax.scatter([p1, p2], [L(p1), L(p2)], color=C_NEG, zorder=4, s=40)
    ax.scatter([pp], [L(pp)], color="k", zorder=4, s=40)
    ax.scatter([pp], [hijos], facecolor="white", edgecolor=C_NEG, zorder=4, s=40, lw=1.5)
    ax.annotate("$R_1$", (p1, L(p1)), xytext=(-14, 6), textcoords="offset points", color=C_NEG)
    ax.annotate("$R_2$", (p2, L(p2)), xytext=(-14, 6), textcoords="offset points", color=C_NEG)
    ax.annotate("$R_p$", (pp, L(pp)), xytext=(-14, 8), textcoords="offset points")
    ax.set_xlabel(r"$\hat p$ (proporción de positivos)")
    ax.set_ylabel(r"$L(\hat p)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08 * max(L(ps)))
    ax.set_title(
        rf"$\hat p_1 = {p1:.2f}$, $\hat p_2 = {p2:.2f}$, "
        rf"$|R_1|/|R_p| = {fraccion_r1:.2f}$ $\Rightarrow$ $\hat p_p = {pp:.2f}$",
        fontsize=10,
    )
    ax.legend(loc="upper left")
    plt.show()


def concavidad():
    """V2. Curva de impureza, cuerda entre dos hijos y ganancia."""
    w = _widgets()
    w.interact(
        dibujar_concavidad,
        p1=_slider(w, "p̂₁", 0, 1, 0.6, 0.01),
        p2=_slider(w, "p̂₂", 0, 1, 1.0, 0.01),
        fraccion_r1=_slider(w, "|R₁| / |Rₚ|", 0.01, 0.99, 0.5, 0.01),
        impureza=w.ToggleButtons(options=list(IMPUREZAS), description="impureza"),
    )


# ---------------------------------------------------------------------------
# V3. Árbol de regresión
# ---------------------------------------------------------------------------


def dibujar_regresion(profundidad=2, min_hoja=1):
    X, y = datos_nieve()
    Xt, yt = datos_nieve(2000, 8)
    xs = np.linspace(1, 13, 1200).reshape(-1, 1)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a = DecisionTreeRegressor(max_depth=profundidad, min_samples_leaf=min_hoja, random_state=0).fit(X, y)
    ax.scatter(X[:, 0], y, s=14, color="0.35", zorder=3)
    ax.plot(xs[:, 0], a.predict(xs), color=C_POS, lw=2)
    ecm_tr = np.mean((a.predict(X) - y) ** 2)
    ecm_te = np.mean((a.predict(Xt) - yt) ** 2)
    ax.set_title(f"{a.get_n_leaves()} hojas · ECM entrenamiento {ecm_tr:.1f} · prueba {ecm_te:.1f}", fontsize=10)
    ax.set_xlabel("mes")
    ax.set_ylabel("cm de nieve por día")
    ax.set_ylim(-1, 28)

    profs = range(1, 13)
    tr, te = [], []
    for p in profs:
        b = DecisionTreeRegressor(max_depth=p, min_samples_leaf=min_hoja, random_state=0).fit(X, y)
        tr.append(np.mean((b.predict(X) - y) ** 2))
        te.append(np.mean((b.predict(Xt) - yt) ** 2))
    ax2.plot(profs, tr, "-o", color=C_NEG, ms=4, label="entrenamiento")
    ax2.plot(profs, te, "-o", color=C_POS, ms=4, label="prueba")
    ax2.axvline(profundidad, color="k", ls=":", lw=1)
    ax2.set_xlabel("profundidad máxima")
    ax2.set_ylabel("error cuadrático medio")
    ax2.set_title(f"Error vs. profundidad (con min_samples_leaf = {min_hoja})", fontsize=10)
    ax2.legend()
    fig.tight_layout()
    plt.show()


def arbol_regresion():
    """V3. Árbol de regresión sobre los datos de nieve."""
    w = _widgets()
    w.interact(
        dibujar_regresion,
        profundidad=_slider_int(w, "max_depth", 1, 12, 2),
        min_hoja=_slider_int(w, "min_samples_leaf", 1, 20, 1),
    )


# ---------------------------------------------------------------------------
# V4. Overfitting y regularización en 2D
# ---------------------------------------------------------------------------

LIM_LUNAS = (-1.8, 2.8, -1.4, 1.9)


def dibujar_regularizacion(profundidad=12, min_hoja=1, ccp_alpha=0.0):
    X, y = datos_lunas()
    Xt, yt = datos_lunas(5000, 3)
    a = DecisionTreeClassifier(max_depth=profundidad, min_samples_leaf=min_hoja, ccp_alpha=ccp_alpha,
                               random_state=0).fit(X, y)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    _fondo(ax, lambda Z: a.predict(Z), LIM_LUNAS)
    _puntos(ax, X, y, tam=16)
    ax.set_title(
        f"{a.get_n_leaves()} hojas · exactitud: entrenamiento {a.score(X, y):.1%}, prueba {a.score(Xt, yt):.1%}",
        fontsize=10,
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    plt.show()


def regularizacion():
    """V4. Regularización de un árbol: profundidad, tamaño de hoja y poda."""
    w = _widgets()
    w.interact(
        dibujar_regularizacion,
        profundidad=_slider_int(w, "max_depth", 1, 15, 12),
        min_hoja=_slider_int(w, "min_samples_leaf", 1, 30, 1),
        ccp_alpha=w.FloatSlider(value=0, min=0, max=0.05, step=0.001, readout_format=".3f",
                                description="ccp_alpha (poda)", continuous_update=False,
                                style={"description_width": "initial"}),
    )


# ---------------------------------------------------------------------------
# V5. Varianza del promedio de modelos correlacionados
# ---------------------------------------------------------------------------


def dibujar_varianza(rho=0.3, B=10, simulaciones=4000):
    rng = np.random.default_rng(0)
    Bs = np.arange(1, 101)

    def promedios(b):
        comun = rng.standard_normal((simulaciones, 1))
        propio = rng.standard_normal((simulaciones, b))
        X = np.sqrt(rho) * comun + np.sqrt(1 - rho) * propio  # varianza 1, correlación rho
        return X.mean(axis=1)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax.plot(Bs, rho + (1 - rho) / Bs, color="k", lw=2, label=r"$\rho\sigma^2 + (1-\rho)\sigma^2/B$")
    ax.axhline(rho, color="gray", ls="--", lw=1, label=r"límite $\rho\sigma^2$")
    bs_sim = [1, 2, 5, 10, 20, 50, 100]
    ax.scatter(bs_sim, [promedios(b).var() for b in bs_sim], color=C_POS, zorder=3, label="simulación")
    ax.axvline(B, color="k", ls=":", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("$B$ (número de modelos)")
    ax.set_ylabel(r"Var$(\bar X)$")
    ax.set_title(rf"$\sigma^2 = 1$, $\rho = {rho:.2f}$", fontsize=10)
    ax.legend()

    bins = np.linspace(-4, 4, 60)
    ax2.hist(promedios(1), bins=bins, alpha=0.5, color=C_NEG, density=True, label="un modelo")
    ax2.hist(promedios(B), bins=bins, alpha=0.6, color=C_POS, density=True, label=f"promedio de B = {B}")
    ax2.set_title("Distribución de las predicciones (simuladas)", fontsize=10)
    ax2.legend()
    fig.tight_layout()
    plt.show()


def varianza_promedio():
    """V5. Var del promedio de B modelos con correlación rho."""
    w = _widgets()
    w.interact(
        dibujar_varianza,
        rho=_slider(w, "ρ (correlación)", 0, 0.95, 0.3, 0.05),
        B=_slider_int(w, "B", 1, 100, 10),
        simulaciones=w.fixed(4000),
    )


# ---------------------------------------------------------------------------
# V6. Bagging y random forest
# ---------------------------------------------------------------------------


def dibujar_bagging(B=50, metodo="bagging", semilla=0):
    X, y = datos_lunas()
    Xt, yt = datos_lunas(5000, 3)
    k = 2 if metodo == "bagging" else 1
    bosque = RandomForestClassifier(n_estimators=B, max_features=k, bootstrap=True, random_state=semilla).fit(X, y)

    fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
    for i in range(3):
        if i < B:
            arbol = bosque.estimators_[i]
            _fondo(axs[i], lambda Z, a=arbol: a.predict(Z), LIM_LUNAS, n=200)
            axs[i].set_title(f"Árbol {i + 1} · prueba {arbol.score(Xt, yt):.1%}", fontsize=10)
        else:
            axs[i].set_axis_off()
        _puntos(axs[i], X, y, tam=8)
    _fondo(axs[3], lambda Z: bosque.predict_proba(Z)[:, 1], LIM_LUNAS, n=200)
    _puntos(axs[3], X, y, tam=8)
    nombre = "Bagging" if metodo == "bagging" else "Random forest"
    axs[3].set_title(f"{nombre}, B = {B} · prueba {bosque.score(Xt, yt):.1%}", fontsize=10)
    fig.tight_layout()
    plt.show()


def bagging():
    """V6. Algunos árboles individuales y su promedio."""
    w = _widgets()
    w.interact(
        dibujar_bagging,
        B=_slider_int(w, "B", 1, 300, 50),
        metodo=w.ToggleButtons(options=["bagging", "random forest"], description="método"),
        semilla=_slider_int(w, "semilla", 0, 20, 0),
    )


# ---------------------------------------------------------------------------
# V7. AdaBoost paso a paso
# ---------------------------------------------------------------------------

_ADA_POS = [(0.8, 3.9), (1.3, 3.0), (2.1, 4.3), (2.6, 3.4), (1.0, 1.4), (3.7, 4.2), (4.3, 3.6), (1.6, 0.6)]
_ADA_NEG = [(3.4, 2.2), (4.2, 1.2), (3.0, 0.8), (4.4, 2.6), (2.2, 2.1), (3.8, 0.5), (0.6, 2.3), (2.6, 1.3)]


@lru_cache(maxsize=4)
def _adaboost(rondas=15):
    X = np.array(_ADA_POS + _ADA_NEG)
    y = np.array([1] * len(_ADA_POS) + [-1] * len(_ADA_NEG))
    w = np.ones(len(y)) / len(y)
    historia = []
    for _ in range(rondas):
        t = DecisionTreeClassifier(max_depth=1).fit(X, y, sample_weight=w)
        mal = t.predict(X) != y
        err = max(w[mal].sum() / w.sum(), 1e-10)
        alpha = np.log((1 - err) / err)
        historia.append((w / w.mean(), t, mal, err, alpha))
        w = w * np.exp(alpha * mal)
        w = w / w.sum()
    return X, y, historia


def dibujar_adaboost(ronda=1):
    X, y, historia = _adaboost()
    pesos, t, mal, err, alpha = historia[ronda - 1]
    y01 = (y > 0).astype(int)
    lim = (0, 5, 0, 5)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    _fondo(ax, lambda Z: (t.predict(Z) > 0).astype(float), lim, n=300)
    _puntos(ax, X, y01, tam=45, pesos=pesos)
    ax.scatter(X[mal, 0], X[mal, 1], s=260, facecolors="none", edgecolors="k", lw=1.2, zorder=4)
    ax.set_title(f"Ronda {ronda}: tocón $G_{{{ronda}}}$ · err = {err:.3f} · α = {alpha:.2f}\n"
                 "(tamaño ∝ peso del ejemplo, círculos = errores)", fontsize=10)

    def voto(Z):
        return sum(a * tt.predict(Z) for _, tt, _, _, a in historia[:ronda])

    errores = int((np.sign(voto(X)) != y).sum())
    _fondo(ax2, lambda Z: (voto(Z) > 0).astype(float), lim, n=300)
    _puntos(ax2, X, y01, tam=45)
    ax2.set_title(f"Votación de los tocones 1 a {ronda}\nerrores de entrenamiento: {errores}", fontsize=10)
    for a in (ax, ax2):
        a.set_xlabel("$x_1$")
        a.set_ylabel("$x_2$")
    fig.tight_layout()
    plt.show()


def adaboost():
    """V7. AdaBoost ronda por ronda."""
    w = _widgets()
    w.interact(dibujar_adaboost, ronda=w.IntSlider(value=1, min=1, max=15, description="ronda b"))


# ---------------------------------------------------------------------------
# V8. Gradient boosting con pérdida cuadrática
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _boosting(nu, profundidad, B_max=200):
    X, y = datos_nieve()
    Xt, yt = datos_nieve(2000, 8)
    xs = np.linspace(1, 13, 600).reshape(-1, 1)
    f0 = y.mean()
    F, Ft, Fs = np.full(len(y), f0), np.full(len(yt), f0), np.full(len(xs), f0)
    estados = [(Fs.copy(), y - F)]  # estado después de b árboles: (curva, residuos)
    arboles, curva = [], []
    for _ in range(B_max):
        h = DecisionTreeRegressor(max_depth=profundidad, random_state=0).fit(X, y - F)
        arboles.append(h)
        F = F + nu * h.predict(X)
        Ft = Ft + nu * h.predict(Xt)
        Fs = Fs + nu * h.predict(xs)
        estados.append((Fs.copy(), y - F))
        curva.append((np.mean((y - F) ** 2), np.mean((yt - Ft) ** 2)))
    return X, y, xs, estados, arboles, np.array(curva)


def dibujar_boosting(B=5, nu=0.3, profundidad=2):
    X, y, xs, estados, arboles, curva = _boosting(round(nu, 3), profundidad)
    Fs, r = estados[B]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    axs[0].scatter(X[:, 0], y, s=12, color="0.35", zorder=3)
    axs[0].plot(xs[:, 0], Fs, color=C_POS, lw=2)
    axs[0].set_title(f"Modelo $F_{{{B}}}$ (con ν = {nu:g}, árboles de profundidad {profundidad})", fontsize=10)
    axs[0].set_ylim(-1, 28)
    axs[0].set_xlabel("mes")

    axs[1].axhline(0, color="gray", lw=0.7)
    axs[1].scatter(X[:, 0], r, s=12, color="0.35", zorder=3)
    if B < len(arboles):
        axs[1].plot(xs[:, 0], arboles[B].predict(xs), color=C_NEG, lw=2, label=f"árbol $h_{{{B + 1}}}$")
        axs[1].legend()
    axs[1].set_title(f"Residuos de $F_{{{B}}}$ y el siguiente árbol", fontsize=10)
    axs[1].set_xlabel("mes")

    bs = np.arange(1, len(curva) + 1)
    axs[2].plot(bs, curva[:, 0], color=C_NEG, label="entrenamiento")
    axs[2].plot(bs, curva[:, 1], color=C_POS, label="prueba")
    axs[2].axvline(max(B, 1), color="k", ls=":", lw=1)
    mejor = int(np.argmin(curva[:, 1])) + 1
    axs[2].scatter([mejor], [curva[mejor - 1, 1]], color=C_POS, zorder=3)
    axs[2].set_ylim(0, 12)
    axs[2].set_xlabel("B")
    axs[2].set_title(f"Error cuadrático medio (mínimo de prueba en B = {mejor})", fontsize=10)
    axs[2].legend()
    fig.tight_layout()
    plt.show()


def gradient_boosting():
    """V8. Boosting que ajusta residuos, árbol por árbol."""
    w = _widgets()
    w.interact(
        dibujar_boosting,
        B=w.IntSlider(value=1, min=0, max=200, description="B"),
        nu=_slider(w, "ν (learning rate)", 0.05, 1.0, 0.3, 0.05),
        profundidad=_slider_int(w, "profundidad", 1, 4, 2),
    )


# ---------------------------------------------------------------------------
# V9. XGBoost: lambda encoge las hojas, gamma poda divisiones
# ---------------------------------------------------------------------------


def dibujar_xgboost(reg_lambda=0.0, gamma=0.0, profundidad=3):
    import xgboost as xgb

    X, y = datos_nieve()
    xs = np.linspace(1, 13, 1200).reshape(-1, 1)
    f0 = float(y.mean())
    r = y - f0

    def un_arbol(lam, gam):
        # Un solo árbol, sumado completo (learning_rate = 1) sobre el promedio
        m = xgb.XGBRegressor(n_estimators=1, learning_rate=1.0, max_depth=profundidad, reg_lambda=lam,
                             gamma=gam, base_score=f0, min_child_weight=0, tree_method="exact")
        return m.fit(X, y)

    base = un_arbol(0.0, 0.0)
    modelo = un_arbol(reg_lambda, gamma)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    ax.scatter(X[:, 0], y, s=12, color="0.35", zorder=3)
    ax.plot(xs[:, 0], base.predict(xs), color="gray", ls="--", lw=1.4, label="λ = 0, γ = 0")
    ax.plot(xs[:, 0], modelo.predict(xs), color=C_POS, lw=2, label=f"λ = {reg_lambda:g}, γ = {gamma:g}")
    ax.axhline(f0, color="k", lw=0.7, ls=":")
    ax.set_ylim(-1, 28)
    ax.set_xlabel("mes")
    ax.set_title("Un árbol de XGBoost sumado al promedio (línea punteada)", fontsize=10)
    ax.legend(loc="upper center")

    # Tabla de hojas: n_j, suma de residuos, valor de la hoja y la fórmula S_j / (n_j + lambda)
    hojas = modelo.apply(X).ravel()
    valores = modelo.predict(X) - f0
    filas = []
    for h in np.unique(hojas):
        sel = hojas == h
        n, S = int(sel.sum()), float(r[sel].sum())
        filas.append([str(n), f"{S / n:.2f}", f"{valores[sel][0]:.2f}", f"{S / (n + reg_lambda):.2f}"])
    ax2.axis("off")
    tabla = ax2.table(cellText=filas, colLabels=["$n_j$", "promedio de $r$", "valor de la hoja",
                                                 r"$S_j/(n_j+\lambda)$"], loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1, 1.3)
    ax2.set_title(f"{len(filas)} hojas (con γ = 0 y esta profundidad serían {len(np.unique(base.apply(X)))})",
                  fontsize=10)
    fig.tight_layout()
    plt.show()


def xgboost_regularizacion():
    """V9. Efecto de lambda (encoge las hojas) y gamma (poda) en un árbol de XGBoost."""
    w = _widgets()
    w.interact(
        dibujar_xgboost,
        reg_lambda=_slider(w, "λ (reg_lambda)", 0, 50, 0, 1),
        gamma=_slider(w, "γ (gamma)", 0, 2000, 0, 25),
        profundidad=_slider_int(w, "max_depth", 1, 4, 3),
    )
