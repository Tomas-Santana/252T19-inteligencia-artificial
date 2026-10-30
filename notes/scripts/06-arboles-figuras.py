"""Genera los datos de las figuras de la clase 6 (árboles de decisión y ensamblaje).

Uso: python generar_figuras.py

Escribe un archivo JSON por figura en la carpeta figs/. Las notas (06-arboles.typ)
los leen con json(...) y dibujan las figuras con cetz.
"""

import json
from pathlib import Path

import numpy as np
from skimage.measure import find_contours
from sklearn.datasets import make_hastie_10_2, make_moons
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

FIGS = Path(__file__).parent / "figs/06"
FIGS.mkdir(exist_ok=True)


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
    print(f"figs/{nombre}.json")


def separar(X, y, pos=1):
    """Devuelve los puntos de cada clase como listas de pares (x1, x2)."""
    return {"neg": X[y != pos].tolist(), "pos": X[y == pos].tolist()}


def rectangulos(arbol, limites):
    """Regiones (hojas) de un árbol entrenado con 2 features.

    Devuelve una lista de rectángulos (x0, x1, y0, y1, prediccion).
    """
    t = arbol.tree_
    salida = []

    def visitar(nodo, x0, x1, y0, y1):
        if t.children_left[nodo] == -1:
            valor = t.value[nodo][0]
            if arbol.__class__.__name__.endswith("Classifier"):
                pred = int(arbol.classes_[np.argmax(valor)])
            else:
                pred = float(valor[0])
            salida.append([x0, x1, y0, y1, pred])
            return
        j, umbral = t.feature[nodo], t.threshold[nodo]
        if j == 0:
            visitar(t.children_left[nodo], x0, umbral, y0, y1)
            visitar(t.children_right[nodo], umbral, x1, y0, y1)
        else:
            visitar(t.children_left[nodo], x0, x1, y0, umbral)
            visitar(t.children_right[nodo], x0, x1, umbral, y1)

    visitar(0, *limites)
    return salida


def contornos(f, limites, n=300, nivel=0.5):
    """Curvas de nivel f(x) = nivel sobre una malla, como listas de puntos."""
    x0, x1, y0, y1 = limites
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    XX, YY = np.meshgrid(xs, ys)
    Z = f(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
    curvas = []
    for c in find_contours(Z, nivel):
        # c tiene coordenadas (fila, columna) en la malla
        px = np.interp(c[:, 1], np.arange(n), xs)
        py = np.interp(c[:, 0], np.arange(n), ys)
        curvas.append(np.c_[px, py][::2].tolist())
    return curvas


def escalera(arbol, x_min, x_max):
    """Función constante a trozos de un árbol de regresión con 1 feature."""
    umbrales = sorted(t for t in arbol.tree_.threshold if t != -2)
    bordes = [x_min] + umbrales + [x_max]
    puntos = []
    for a, b in zip(bordes[:-1], bordes[1:]):
        v = float(arbol.predict([[(a + b) / 2]])[0])
        puntos += [[a, v], [b, v]]
    return puntos


def escalera_f(f, x_min, x_max, n=1200):
    xs = np.linspace(x_min, x_max, n)
    return np.c_[xs, f(xs.reshape(-1, 1))].tolist()


# ---------------------------------------------------------------------------
# 1. Ejemplo del esquí: mes del año vs. latitud
# ---------------------------------------------------------------------------


def esquiable(mes, lat):
    norte = (lat > 30) & ((mes < 3) | (mes >= 12))
    sur = (lat < -30) & (mes >= 6) & (mes < 9)
    return (norte | sur).astype(int)


def figura_esqui():
    rng = np.random.default_rng(3)
    m = 140
    mes = rng.uniform(1, 13, m)
    lat = rng.uniform(-90, 90, m)
    X = np.c_[mes, lat]
    y = esquiable(mes, lat)

    # Las regiones del árbol de la clase se dibujan directamente en las notas
    # (son las de la regla `esquiable`). Aquí solo guardamos los puntos y
    # los conteos de las dos primeras preguntas: latitud > 30 y, arriba, mes < 3.
    arriba = X[:, 1] > 30
    arriba_izq = arriba & (X[:, 0] < 3)
    datos = {
        "m": m,
        **separar(X, y),
        "n_pos": int(y.sum()),
        "arriba": [int(y[arriba].sum()), int((1 - y[arriba]).sum())],
        "abajo": [int(y[~arriba].sum()), int((1 - y[~arriba]).sum())],
        "arriba_izq": [int(y[arriba_izq].sum()), int((1 - y[arriba_izq]).sum())],
    }
    guardar("esqui", datos)


# ---------------------------------------------------------------------------
# 2. Árbol de regresión: nieve por día según el mes (hemisferio norte)
#    Los mismos datos se usan después en gradient boosting.
# ---------------------------------------------------------------------------


def datos_nieve(m, semilla):
    rng = np.random.default_rng(semilla)
    x = np.sort(rng.uniform(1, 13, m))
    verdad = 20 * np.maximum(0, np.cos(2 * np.pi * (x - 1) / 12)) ** 1.5
    y = np.maximum(0, verdad + rng.normal(0, 2.5, m))
    return x, y


def figura_regresion():
    x, y = datos_nieve(80, 7)
    X = x.reshape(-1, 1)
    paneles = {}
    for nombre, prof in (("p1", 1), ("p2", 2), ("p8", None)):
        a = DecisionTreeRegressor(max_depth=prof, random_state=0).fit(X, y)
        paneles[nombre] = {
            "prof": prof if prof else int(a.get_depth()),
            "hojas": int(a.get_n_leaves()),
            "f": escalera(a, 1, 13),
            "ecm": float(np.mean((a.predict(X) - y) ** 2)),
        }
    guardar("regresion", {"m": len(x), "puntos": np.c_[x, y].tolist(), "paneles": paneles})


# Regiones del árbol del esquí: (mes0, mes1, lat0, lat1, se puede esquiar)
REGIONES_ESQUI = [
    (1, 3, 30, 90, 1),
    (3, 12, 30, 90, 0),
    (12, 13, 30, 90, 1),
    (1, 13, -30, 30, 0),
    (1, 6, -90, -30, 0),
    (6, 9, -90, -30, 1),
    (9, 13, -90, -30, 0),
]


def figura_esqui_regresion():
    """El plano del esquí con cm de nieve por día en lugar de +/-.

    Se usan pocos puntos, separados entre sí, para que los números se puedan leer.
    """
    rng = np.random.default_rng(11)
    puntos, regiones = [], []
    for x0, x1, y0, y1, nieva in REGIONES_ESQUI:
        area = (x1 - x0) / 12 * (y1 - y0) / 180
        n = max(6 if nieva else 3, round(40 * area))
        elegidos = []
        intentos = 0
        while len(elegidos) < n and intentos < 5000:
            intentos += 1
            # margen para que el número no toque las líneas de la región
            mx, my = 0.3, 9
            p = (rng.uniform(x0 + mx, x1 - mx), rng.uniform(y0 + my, y1 - my))
            todos = elegidos + [q[:2] for q in puntos]
            if all(((p[0] - q[0]) / 12) ** 2 + ((p[1] - q[1]) / 180) ** 2 > 0.07**2 for q in todos):
                elegidos.append(p)
        valores = rng.integers(5, 16, len(elegidos)) if nieva else np.zeros(len(elegidos), dtype=int)
        puntos += [[px, py, int(v)] for (px, py), v in zip(elegidos, valores)]
        regiones.append([x0, x1, y0, y1, float(np.mean(valores))])
    guardar("esqui_regresion", {"m": len(puntos), "puntos": puntos, "regiones": regiones})


# ---------------------------------------------------------------------------
# 2b. Matriz de confusión: esquí con 400 ejemplos (los mismos del laboratorio)
# ---------------------------------------------------------------------------


def figura_confusion():
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(3)
    m = 400
    mes = rng.uniform(1, 13, m)
    lat = rng.uniform(-90, 90, m)
    X = np.c_[mes, lat]
    y = esquiable(mes, lat)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    arbol = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_tr, y_tr)
    datos = {"m_test": len(y_te), "pos_test": int(y_te.sum())}
    for nombre, pred in (("siempre_no", np.zeros_like(y_te)), ("arbol", arbol.predict(X_te))):
        # filas: clase real (0, 1); columnas: clase predicha (0, 1)
        datos[nombre] = confusion_matrix(y_te, pred, labels=[0, 1]).tolist()
    guardar("confusion", datos)


# ---------------------------------------------------------------------------
# 3. XOR: una primera pregunta "inútil" que hace posible una segunda perfecta
# ---------------------------------------------------------------------------


def figura_xor():
    rng = np.random.default_rng(1)
    centros = [(1, 1, 0), (3, 3, 0), (1, 3, 1), (3, 1, 1)]
    X, y = [], []
    for cx, cy, c in centros:
        pts = rng.uniform(-0.75, 0.75, (12, 2)) + [cx, cy]
        X.append(pts)
        y += [c] * 12
    X = np.vstack(X)
    y = np.array(y)

    def gini(yy):
        p = yy.mean()
        return 2 * p * (1 - p)

    # Ganancia de la mejor división sobre x1 (o x2) en el padre
    mejor = 0.0
    for j in (0, 1):
        for t in np.unique(X[:, j]):
            izq = X[:, j] < t
            if izq.all() or (~izq).all():
                continue
            g = gini(y) - (izq.mean() * gini(y[izq]) + (~izq).mean() * gini(y[~izq]))
            mejor = max(mejor, g)
    guardar("xor", {**separar(X, y), "ganancia_max": mejor, "gini_padre": gini(y)})


# ---------------------------------------------------------------------------
# 4. Estructura aditiva: frontera diagonal x2 = x1
# ---------------------------------------------------------------------------


def figura_diagonal():
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 5, (120, 2))
    y = (X[:, 1] > X[:, 0]).astype(int)
    arbol = DecisionTreeClassifier(random_state=0).fit(X, y)
    guardar(
        "diagonal",
        {
            **separar(X, y),
            "hojas": rectangulos(arbol, (0, 5, 0, 5)),
            "n_hojas": int(arbol.get_n_leaves()),
        },
    )


# ---------------------------------------------------------------------------
# 5. Bootstrap: tres muestras de un conjunto de 10 ejemplos
# ---------------------------------------------------------------------------


def figura_bootstrap():
    rng = np.random.default_rng(5)
    m = 10
    muestras = [sorted(rng.integers(1, m + 1, m).tolist()) for _ in range(3)]
    unicos = [len(set(z)) for z in muestras]
    # Fracción esperada de ejemplos distintos, para varios m
    fraccion = {str(n): 1 - (1 - 1 / n) ** n for n in (2, 5, 10, 100, 1000)}
    guardar("bootstrap", {"m": m, "muestras": muestras, "unicos": unicos, "fraccion": fraccion})


# ---------------------------------------------------------------------------
# 6. Un árbol, bagging y random forest sobre los mismos datos
# ---------------------------------------------------------------------------


def figura_bagging():
    X, y = make_moons(n_samples=200, noise=0.35, random_state=2)
    Xt, yt = make_moons(n_samples=5000, noise=0.35, random_state=3)
    lim = (-1.8, 2.8, -1.4, 1.9)
    B = 300

    modelos = {
        "arbol": DecisionTreeClassifier(random_state=0),
        "bagging": BaggingClassifier(DecisionTreeClassifier(), n_estimators=B, random_state=0),
        "rf": RandomForestClassifier(n_estimators=B, max_features=1, random_state=0),
    }
    paneles = {}
    for nombre, mod in modelos.items():
        mod.fit(X, y)
        paneles[nombre] = {
            "frontera": contornos(lambda Z: mod.predict_proba(Z)[:, 1], lim),
            "exactitud_train": mod.score(X, y),
            "exactitud_test": mod.score(Xt, yt),
        }
    guardar("bagging", {"m": len(y), "B": B, **separar(X, y), "paneles": paneles})


# ---------------------------------------------------------------------------
# 7. Error de prueba vs. número de árboles (datos de Hastie et al., 10 features)
# ---------------------------------------------------------------------------


def figura_error_vs_B():
    X, y = make_hastie_10_2(n_samples=12000, random_state=1)
    X, Xt, y, yt = X[:2000], X[2000:], y[:2000], y[2000:]
    B = 400
    pasos = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300, 350, 400]

    def curva_promedio(mod):
        mod.fit(X, y)
        acum = np.zeros(len(yt))
        errores = []
        for b, est in enumerate(mod.estimators_, start=1):
            if hasattr(mod, "estimators_features_"):
                cols = mod.estimators_features_[b - 1]
                acum += est.predict_proba(Xt[:, cols])[:, 1]
            else:
                acum += est.predict_proba(Xt)[:, 1]
            if b in pasos:
                pred = np.where(acum / b > 0.5, 1, -1)
                errores.append([b, float(np.mean(pred != yt))])
        return errores

    bag = curva_promedio(BaggingClassifier(DecisionTreeClassifier(), n_estimators=B, random_state=0))
    rf = curva_promedio(RandomForestClassifier(n_estimators=B, max_features=3, random_state=0))

    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=B, random_state=0)
    ada.fit(X, y)
    err_ada = []
    for b, pred in enumerate(ada.staged_predict(Xt), start=1):
        if b in pasos:
            err_ada.append([b, float(np.mean(pred != yt))])

    arbol = DecisionTreeClassifier(random_state=0).fit(X, y)
    tocon = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
    guardar(
        "error_vs_B",
        {
            "m": len(y),
            "m_test": len(yt),
            "bagging": bag,
            "rf": rf,
            "adaboost": err_ada,
            "arbol": float(1 - arbol.score(Xt, yt)),
            "tocon": float(1 - tocon.score(Xt, yt)),
        },
    )


# ---------------------------------------------------------------------------
# 8. AdaBoost paso a paso con tocones (decision stumps)
# ---------------------------------------------------------------------------


def figura_adaboost():
    pos = [(0.8, 3.9), (1.3, 3.0), (2.1, 4.3), (2.6, 3.4), (1.0, 1.4), (3.7, 4.2), (4.3, 3.6), (1.6, 0.6)]
    neg = [(3.4, 2.2), (4.2, 1.2), (3.0, 0.8), (4.4, 2.6), (2.2, 2.1), (3.8, 0.5), (0.6, 2.3), (2.6, 1.3)]
    X = np.array(pos + neg)
    y = np.array([1] * len(pos) + [-1] * len(neg))
    m = len(y)
    w = np.ones(m) / m
    rondas = []
    F = lambda Z: np.zeros(len(Z))  # noqa: E731
    tocones = []
    for _ in range(3):
        t = DecisionTreeClassifier(max_depth=1).fit(X, y, sample_weight=w)
        pred = t.predict(X)
        mal = pred != y
        err = w[mal].sum() / w.sum()
        alpha = np.log((1 - err) / err)
        rondas.append(
            {
                "pesos": (w / w.mean()).tolist(),  # peso relativo (1 = peso inicial)
                "feature": int(t.tree_.feature[0]),
                "umbral": float(t.tree_.threshold[0]),
                # clase predicha a la izquierda / abajo del umbral
                "izq": int(t.classes_[np.argmax(t.tree_.value[t.tree_.children_left[0]][0])]),
                "mal": np.where(mal)[0].tolist(),
                "err": err,
                "alpha": alpha,
            }
        )
        tocones.append((t, alpha))
        w = w * np.exp(alpha * mal)
        w = w / w.sum()

    def G(Z):
        return sum(a * t.predict(Z) for t, a in tocones)

    final_mal = np.where(np.sign(G(X)) != y)[0].tolist()

    # Regiones de la votación final: rectángulos entre los umbrales de los tocones
    ux = sorted({0.0, 5.0} | {r["umbral"] for r in rondas if r["feature"] == 0})
    uy = sorted({0.0, 5.0} | {r["umbral"] for r in rondas if r["feature"] == 1})
    regiones = []
    for x0, x1 in zip(ux[:-1], ux[1:]):
        for y0, y1 in zip(uy[:-1], uy[1:]):
            clase = int(G(np.array([[(x0 + x1) / 2, (y0 + y1) / 2]]))[0] > 0)
            regiones.append([x0, x1, y0, y1, clase])
    guardar(
        "adaboost",
        {
            "puntos": X.tolist(),
            "y": y.tolist(),
            "rondas": rondas,
            "frontera": contornos(G, (0, 5, 0, 5), n=400, nivel=0.0),
            "final_mal": final_mal,
            "regiones": regiones,
        },
    )


# ---------------------------------------------------------------------------
# 9. Boosting con pérdida cuadrática: ajustar los residuos
# ---------------------------------------------------------------------------


def figura_gradient_boosting():
    x, y = datos_nieve(80, 7)
    xt, yt = datos_nieve(2000, 8)
    X, Xt = x.reshape(-1, 1), xt.reshape(-1, 1)
    nu = 0.3
    prof = 2

    f0 = float(y.mean())
    pred = np.full(len(y), f0)
    pred_t = np.full(len(yt), f0)
    arboles = []
    curva = []
    instantes = {}
    B = 300
    for b in range(1, B + 1):
        r = y - pred
        h = DecisionTreeRegressor(max_depth=prof, random_state=0).fit(X, r)
        arboles.append(h)
        if b == 1:
            primer = {"residuos": np.c_[x, r].tolist(), "h": escalera(h, 1, 13)}
        pred = pred + nu * h.predict(X)
        pred_t = pred_t + nu * h.predict(Xt)
        curva.append([b, float(np.mean((y - pred) ** 2)), float(np.mean((yt - pred_t) ** 2))])
        if b in (1, 3, 10, 50):

            def f(Z, k=b):
                return f0 + nu * sum(a.predict(Z) for a in arboles[:k])

            instantes[str(b)] = escalera_f(f, 1, 13)

    guardar(
        "boosting",
        {
            "puntos": np.c_[x, y].tolist(),
            "f0": f0,
            "nu": nu,
            "prof": prof,
            "primer": primer,
            "instantes": instantes,
            "curva": [c for c in curva if c[0] in (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 250, 300)],
        },
    )


if __name__ == "__main__":
    figura_esqui()
    figura_regresion()
    figura_esqui_regresion()
    figura_confusion()
    figura_xor()
    figura_diagonal()
    figura_bootstrap()
    figura_bagging()
    figura_error_vs_B()
    figura_adaboost()
    figura_gradient_boosting()
