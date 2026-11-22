"""Clase 9, laboratorio: funciones de apoyo. No hace falta modificar este archivo.

- Datos: example_1d, make_blob_data, load_wine_data, load_photo, load_shapes.
  Ninguno necesita internet: son sintéticos o vienen incluidos en scikit-learn.
- kmeans: el algoritmo completo de las notas. En cada iteración llama a
  assign_clusters, update_centroids y distortion de agrupamiento.py.
- check_*: verifican las funciones de agrupamiento.py.
- Gráficas: plot_clusters, plot_kmeans_steps, plot_J, plot_seeds,
  elbow_silhouette, show_images, k_distance_plot, run_comparison, plot_comparison.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image, load_wine, make_blobs, make_circles, make_moons
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
MARKERS = ["o", "^", "s", "D", "v", "P", "X", "*"]


def verify(condition, success_message, error_message):
    print(("✅ " + success_message) if condition else ("❌ " + error_message))
    return bool(condition)


def _call(function, *args):
    """Llama a una función de agrupamiento.py. Si falta completarla, avisa y devuelve None."""
    try:
        return function(*args)
    except NotImplementedError as e:
        print(f"❌ {e}")
        return None


def run(function, *args):
    """Llama a una función de agrupamiento.py; si falta completarla, avisa y devuelve None."""
    return _call(function, *args)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


def example_1d():
    """Los datos del Ejercicio 1 de las notas: x = 1, 2, 4, 7, 8, 10, como matriz (6, 1)."""
    return np.array([[1.0], [2.0], [4.0], [7.0], [8.0], [10.0]])


def make_blob_data():
    """300 puntos en el plano, en cuatro grupos. Las etiquetas no se usan."""
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=3)
    return X


def load_wine_data():
    """178 vinos de tres variedades (cultivares), con 13 features químicas.
    Devuelve X (178, 13), y (la variedad: 0, 1 o 2) y los nombres de las features.
    La variedad NO se usa para agrupar: solo para comparar al final."""
    wine = load_wine()
    return wine.data, wine.target, wine.feature_names


def load_photo():
    """Una foto de 427 x 640 píxeles, como arreglo (alto, ancho, 3) con valores entre 0 y 1."""
    return load_sample_image("china.jpg") / 255


def load_shapes():
    """Tres conjuntos de datos en el plano, con las features estandarizadas.
    Devuelve un diccionario {nombre: (X, número de grupos)}."""
    shapes = {
        "bolas": (make_blobs(n_samples=300, centers=3, cluster_std=1.3, random_state=8)[0], 3),
        "lunas": (make_moons(n_samples=300, noise=0.06, random_state=0)[0], 2),
        "círculos": (make_circles(n_samples=300, factor=0.45, noise=0.05, random_state=0)[0], 2),
    }
    return {name: (StandardScaler().fit_transform(X), k) for name, (X, k) in shapes.items()}


# ---------------------------------------------------------------------------
# K-means con las funciones de agrupamiento.py
# ---------------------------------------------------------------------------


def kmeans(X, k, seed=0, max_iter=100, verbose=False):
    """K-means como en las notas: inicializar los centroides con k ejemplos al azar
    y repetir asignar y mover hasta que nada cambie.

    Devuelve un diccionario con:
      "c": el grupo de cada ejemplo,
      "centroids": los centroides finales,
      "J": la distorsión después de cada iteración,
      "history": la lista de pasos (para graficar).
    Devuelve None si falta completar alguna función de agrupamiento.py.
    """
    import agrupamiento  # aquí y no arriba, para usar siempre la versión más reciente

    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    centroids = X[rng.choice(len(X), k, replace=False)].copy()
    history = [{"step": "inicio", "c": None, "centroids": centroids.copy()}]
    J = []
    c = None
    for iteration in range(max_iter):
        new_c = _call(agrupamiento.assign_clusters, X, centroids)
        if new_c is None:
            return None
        new_c = np.asarray(new_c)
        history.append({"step": "asignar", "c": new_c.copy(), "centroids": centroids.copy()})
        if c is not None and np.array_equal(new_c, c):
            break  # nada cambió: terminar
        c = new_c

        before = centroids
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore")
            centroids = _call(agrupamiento.update_centroids, X, c, k)
        if centroids is None:
            return None
        centroids = np.asarray(centroids, dtype=float)
        empty = np.isnan(centroids).any(axis=1)
        centroids[empty] = before[empty]  # un grupo vacío conserva su centroide
        history.append({"step": "mover", "c": c.copy(), "centroids": centroids.copy(), "before": before.copy()})

        value = _call(agrupamiento.distortion, X, c, centroids)
        if value is None:
            return None
        J.append(float(value))
        if verbose:
            print(f"iteración {iteration + 1}: J = {J[-1]:.1f}")
    if verbose:
        print(f"terminó después de {len(J)} iteraciones")
    return {"c": c, "centroids": centroids, "J": J, "history": history}


# ---------------------------------------------------------------------------
# Verificaciones
# ---------------------------------------------------------------------------


def check_assign_clusters(assign_clusters):
    X = example_1d()
    c = _call(assign_clusters, X, np.array([[1.0], [2.0]]))
    if c is None:
        return
    c = np.asarray(c)
    if not verify(c.shape == (6,), "devuelve un vector de forma (n,)", f"debería devolver forma (6,), y devuelve {c.shape}"):
        return
    verify(np.array_equal(c, [0, 1, 1, 1, 1, 1]),
           "Ejercicio 1, primera asignación: grupos {1} y {2, 4, 7, 8, 10}",
           f"con centroides 1 y 2, los grupos deberían ser [0 1 1 1 1 1], y son {c}")
    rng = np.random.default_rng(0)
    X2, mu = rng.normal(size=(50, 3)), rng.normal(size=(4, 3))
    expected = ((X2[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    c2 = np.asarray(assign_clusters(X2, mu))
    verify(np.array_equal(c2, expected), "funciona con varias features y varios centroides",
           "con datos de 3 features y 4 centroides, algunos ejemplos no van a su centroide más cercano")


def check_update_centroids(update_centroids):
    X = example_1d()
    mu = _call(update_centroids, X, np.array([0, 1, 1, 1, 1, 1]), 2)
    if mu is None:
        return
    mu = np.asarray(mu)
    if not verify(mu.shape == (2, 1), "devuelve una matriz de forma (k, d)", f"debería devolver forma (2, 1), y devuelve {mu.shape}"):
        return
    verify(np.allclose(mu.ravel(), [1, 6.2]),
           "Ejercicio 1, primer paso de mover: centroides 1 y 6.2",
           f"los centroides deberían ser 1 y 6.2, y son {mu.ravel()}")
    rng = np.random.default_rng(0)
    X2, c2 = rng.normal(size=(50, 3)), np.arange(50) % 4
    expected = np.array([X2[c2 == j].mean(axis=0) for j in range(4)])
    verify(np.allclose(np.asarray(update_centroids(X2, c2, 4)), expected), "funciona con varias features y varios grupos",
           "con datos de 3 features y 4 grupos, los centroides no son los promedios")


def check_distortion(distortion):
    X = example_1d()
    J = _call(distortion, X, np.array([0, 1, 1, 1, 1, 1]), np.array([[1.0], [6.2]]))
    if J is None:
        return
    verify(np.isclose(float(J), 40.8), "Ejercicio 1, primera iteración: J = 40.8", f"J debería ser 40.8, y es {J}")
    J = distortion(X, np.array([0, 0, 0, 1, 1, 1]), np.array([[7 / 3], [25 / 3]]))
    verify(np.isclose(float(J), 9.3333, atol=1e-3), "Ejercicio 1, resultado final: J = 9.33", f"J debería ser 9.33, y es {J}")


def check_standardize(standardize):
    X, _, _ = load_wine_data()
    Z = _call(standardize, X)
    if Z is None:
        return
    Z = np.asarray(Z)
    if not verify(Z.shape == X.shape, "devuelve una matriz de la misma forma", f"debería devolver forma {X.shape}, y devuelve {Z.shape}"):
        return
    verify(np.allclose(Z.mean(axis=0), 0), "cada columna tiene media 0", "las columnas no tienen media 0: ¿se resta la media de cada columna (axis=0)?")
    verify(np.allclose(Z.std(axis=0), 1), "cada columna tiene desviación estándar 1",
           "las columnas no tienen desviación 1: ¿se divide entre la desviación de cada columna (axis=0)?")
    verify(np.allclose(Z, StandardScaler().fit_transform(X)), "igual que StandardScaler de scikit-learn",
           "el resultado no coincide con StandardScaler")


def check_compare_algorithms(compare_algorithms):
    X, k = load_shapes()["lunas"]
    result = _call(compare_algorithms, X, k, 0.3)
    if result is None:
        return
    if not verify(set(result) == {"kmeans", "dbscan", "single"}, "tiene las tres claves",
                  f"las claves deberían ser kmeans, dbscan y single, y son {sorted(result)}"):
        return
    verify(all(np.asarray(v).shape == (len(X),) for v in result.values()), "cada método da un grupo por ejemplo",
           "cada valor debería ser un vector con un grupo por ejemplo")
    verify(len(set(result["kmeans"])) == k and len(set(result["single"])) == k, f"K-means y el jerárquico encuentran {k} grupos",
           f"K-means y el jerárquico deberían encontrar {k} grupos: ¿se usa n_clusters=k?")


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------


def plot_clusters(X, c=None, centroids=None, title=None, ax=None, size=12):
    """Los puntos de X (dos features), con un color por grupo. El ruido (-1) va en gris."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    X = np.asarray(X)
    if c is None:
        ax.scatter(X[:, 0], X[:, 1], s=size, color="0.35")
    else:
        c = np.asarray(c)
        for j in sorted(set(c)):
            mask = c == j
            if j == -1:
                ax.scatter(X[mask, 0], X[mask, 1], s=size * 1.3, color="0.55", marker="x", linewidths=1)
            else:
                ax.scatter(X[mask, 0], X[mask, 1], s=size, color=COLORS[j % 8], marker=MARKERS[j % 8])
    if centroids is not None:
        for j, m in enumerate(centroids):
            ax.scatter(*m[:2], s=260, marker="X", color=COLORS[j % 8], edgecolors="white", linewidths=1.8, zorder=3)
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_kmeans_steps(X, result, max_panels=12):
    """Los primeros pasos de K-means: inicio, asignar, mover, asignar, ..."""
    if result is None:
        print("❌ Primero hay que completar las funciones de la Parte 1.")
        return
    steps = result["history"][:max_panels]
    cols = 4
    rows = int(np.ceil(len(steps) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.9 * rows))
    for ax in np.ravel(axs)[len(steps):]:
        ax.axis("off")
    J = iter(result["J"])
    for i, (ax, step) in enumerate(zip(np.ravel(axs), steps)):
        title = f"{i}. {step['step']}"
        if step["step"] == "mover":
            title += f"  (J = {next(J):.0f})"
        plot_clusters(X, step["c"], step["centroids"], title=title, ax=ax, size=8)
        if step["step"] == "mover":
            for a, b in zip(step["before"], step["centroids"]):
                ax.annotate("", xy=b[:2], xytext=a[:2], arrowprops=dict(arrowstyle="->", lw=1.2))
    plt.tight_layout()
    plt.show()


def plot_J(result):
    """La distorsión J después de cada iteración."""
    if result is None:
        print("❌ Primero hay que completar las funciones de la Parte 1.")
        return
    J = result["J"]
    plt.figure(figsize=(5, 3))
    plt.plot(range(1, len(J) + 1), J, "o-")
    plt.xlabel("iteración")
    plt.ylabel("J")
    plt.title("La distorsión en cada iteración")
    plt.gca().xaxis.get_major_locator().set_params(integer=True)
    plt.tight_layout()
    plt.show()


def plot_seeds(X, k, seeds):
    """K-means con varias inicializaciones al azar (una por semilla), con la J final de cada una."""
    results = []
    for s in seeds:
        results.append(kmeans(X, k, seed=s))
        if results[-1] is None:
            print("❌ Primero hay que completar las funciones de la Parte 1.")
            return None
    cols = 4
    rows = int(np.ceil(len(seeds) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.9 * rows))
    for ax in np.ravel(axs)[len(seeds):]:
        ax.axis("off")
    for ax, s, r in zip(np.ravel(axs), seeds, results):
        plot_clusters(X, r["c"], r["centroids"], title=f"semilla {s}: J = {r['J'][-1]:.0f}", ax=ax, size=8)
    plt.tight_layout()
    plt.show()
    return [r["J"][-1] for r in results]


def elbow_silhouette(X, ks=range(2, 9), title=None):
    """Corre KMeans de scikit-learn para cada k, imprime J y la silueta y las grafica."""
    J, S = [], []
    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        J.append(model.inertia_)
        S.append(silhouette_score(X, model.labels_))
        print(f"k={k}  J={model.inertia_:12.1f}  silueta={S[-1]:.3f}")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.3))
    axs[0].plot(list(ks), J, "o-")
    axs[0].set(xlabel="k", ylabel="J", title="Método del codo")
    axs[1].plot(list(ks), S, "o-")
    axs[1].set(xlabel="k", ylabel="silueta", title="Silueta")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def compress(image, k, sample_size=5000, seed=0):
    """Reduce la foto a k colores con el kmeans de este laboratorio.
    K-means se entrena con una muestra de píxeles; después cada píxel toma el color
    de su centroide más cercano. Devuelve la foto comprimida y los k colores."""
    import agrupamiento

    pixels = image.reshape(-1, 3)
    sample = pixels[np.random.default_rng(seed).choice(len(pixels), sample_size, replace=False)]
    result = kmeans(sample, k, seed=seed)
    if result is None:
        return None, None
    labels = np.asarray(agrupamiento.assign_clusters(pixels, result["centroids"]))
    return result["centroids"][labels].reshape(image.shape), result["centroids"]


def show_images(images):
    """images: diccionario {título: (foto, colores o None)}."""
    images = {t: v for t, v in images.items() if v[0] is not None}
    fig, axs = plt.subplots(2, len(images), figsize=(3.6 * len(images), 3.1), gridspec_kw={"height_ratios": [6, 1]})
    axs = np.asarray(axs).reshape(2, -1)
    for (title, (image, palette)), top, bottom in zip(images.items(), axs[0], axs[1]):
        top.imshow(np.clip(image, 0, 1))
        top.set_title(title, fontsize=10)
        top.axis("off")
        bottom.axis("off")
        if palette is not None:
            bottom.imshow(np.clip(palette, 0, 1)[None, :, :], aspect="auto")
    plt.tight_layout()
    plt.show()


def k_distance_plot(X, min_samples=5, eps=None, title=None, ax=None):
    """La distancia de cada punto a su vecino número min_samples (contándose a sí mismo),
    ordenada de menor a mayor. Se escoge eps donde la curva empieza a subir rápido."""
    distances, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.sort(distances[:, -1]))
    if eps is not None:
        ax.axhline(eps, color="#d62728", ls="--", lw=1)
    ax.set(xlabel="puntos, ordenados por distancia", ylabel=f"distancia al {min_samples}.º vecino")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def run_comparison(shapes, eps, min_samples=5):
    """Llama a compare_algorithms de agrupamiento.py con cada conjunto de datos.
    eps: diccionario {nombre: eps}. Devuelve {nombre: resultado}."""
    import agrupamiento

    results = {}
    for name, (X, k) in shapes.items():
        results[name] = _call(agrupamiento.compare_algorithms, X, k, eps[name], min_samples)
        if results[name] is None:
            break
    return results


def plot_comparison(shapes, results):
    """Una fila por conjunto de datos y una columna por método.
    shapes: lo que devuelve load_shapes(); results: {nombre: lo que devuelve compare_algorithms}."""
    names = [n for n in shapes if results.get(n) is not None]
    if not names:
        print("❌ Todavía no hay resultados que graficar.")
        return
    methods = [("kmeans", "K-means"), ("dbscan", "DBSCAN"), ("single", "Jerárquico (single)")]
    fig, axs = plt.subplots(len(names), 3, figsize=(10, 3 * len(names)))
    axs = np.asarray(axs).reshape(len(names), 3)
    for row, name in zip(axs, names):
        X, _ = shapes[name]
        for ax, (key, title) in zip(row, methods):
            c = np.asarray(results[name][key])
            n_groups = len(set(c) - {-1})
            noise = int((c == -1).sum())
            subtitle = f"{n_groups} grupos" + (f", {noise} de ruido" if noise else "")
            plot_clusters(X, c, title=f"{title}\n{name}: {subtitle}", ax=ax, size=7)
    plt.tight_layout()
    plt.show()
