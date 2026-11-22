"""Genera los datos de las figuras y las salidas de código de la clase 9 (agrupamiento).

Uso: python notes/scripts/09-agrupamiento-figuras.py

1. Ejecuta cada fragmento de notes/code/09/ y guarda lo que imprime en
   notes/figs/09/salidas.json.
2. Calcula los datos de las figuras y los guarda en notes/figs/09/datos.json.
3. Guarda la foto original y comprimida en notes/figs/09/*.png.

No necesita internet: los datos son sintéticos o vienen incluidos en scikit-learn.
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import load_sample_image, make_blobs, make_circles, make_moons
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

NOTAS = Path(__file__).parent.parent
CODIGO = NOTAS / "code/09"
FIGS = NOTAS / "figs/09"
FIGS.mkdir(parents=True, exist_ok=True)


def limpiar(x):
    if isinstance(x, dict):
        return {k: limpiar(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [limpiar(v) for v in x]
    if isinstance(x, np.ndarray):
        return limpiar(x.tolist())
    if isinstance(x, (np.floating, float)):
        return round(float(x), 3)
    if isinstance(x, (np.integer, np.bool_)):
        return int(x)
    return x


datos = {}

# ---------------------------------------------------------------------------
# 1. Salidas de los fragmentos de código
# ---------------------------------------------------------------------------
salidas = {}
for archivo in sorted(CODIGO.glob("*.py")):
    r = subprocess.run([sys.executable, archivo.name], cwd=CODIGO, capture_output=True, text=True, check=True)
    salidas[archivo.stem] = r.stdout.rstrip("\n")
with open(FIGS / "salidas.json", "w") as f:
    json.dump(salidas, f, ensure_ascii=False)
print("figs/09/salidas.json")

# ---------------------------------------------------------------------------
# 2. K-means paso a paso (como la figura 10.1 de las notas de CS229)
# ---------------------------------------------------------------------------


def asignar(X, mu):
    return np.argmin(((X[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2), axis=1)


def distorsion(X, c, mu):
    return float(((X - mu[c]) ** 2).sum())


X, _ = make_blobs(n_samples=60, centers=[(-2, 2), (2, 2.5), (0, -1.5)], cluster_std=0.6, random_state=3)
mu = np.array([[1.0, -1.1], [-3.2, -2.9], [2.2, 3.4]])  # centroides iniciales al azar
pasos = [{"tipo": "datos", "c": None, "mu": None}, {"tipo": "inicio", "c": None, "mu": mu.copy()}]
J = []
for it in range(10):
    c = asignar(X, mu)
    pasos.append({"tipo": "asignar", "c": c.copy(), "mu": mu.copy(), "J": distorsion(X, c, mu)})
    J.append(distorsion(X, c, mu))
    nuevo = np.array([X[c == j].mean(axis=0) for j in range(len(mu))])
    pasos.append({"tipo": "mover", "c": c.copy(), "mu": nuevo.copy(), "antes": mu.copy(), "J": distorsion(X, c, nuevo)})
    J.append(distorsion(X, c, nuevo))
    if np.allclose(nuevo, mu):
        break
    mu = nuevo
datos["pasos"] = {"X": X, "pasos": pasos, "J": J, "iteraciones": it + 1}
print("K-means paso a paso:", it + 1, "iteraciones, J:", [round(j, 1) for j in J])

# ---------------------------------------------------------------------------
# 3. Mínimos locales: la misma base de datos, dos inicializaciones
# ---------------------------------------------------------------------------
X, _ = make_blobs(n_samples=200, centers=[(-4, 0), (0, 4), (4, 0), (0, -4)], cluster_std=0.8, random_state=1)
resultados = []
for semilla in range(60):
    km = KMeans(n_clusters=4, init="random", n_init=1, random_state=semilla).fit(X)
    resultados.append((km.inertia_, semilla, km))
resultados.sort(key=lambda t: t[0])
bueno, malo = resultados[0][2], resultados[-1][2]
datos["minimos"] = {
    "X": X,
    "bueno": {"c": bueno.labels_, "mu": bueno.cluster_centers_, "J": bueno.inertia_},
    "malo": {"c": malo.labels_, "mu": malo.cluster_centers_, "J": malo.inertia_},
}
print("mínimos locales: J bueno", round(bueno.inertia_, 1), "J malo", round(malo.inertia_, 1))

# ---------------------------------------------------------------------------
# 4. Elección de k (los mismos datos de code/09/elegir_k.py)
# ---------------------------------------------------------------------------
X, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.9, random_state=7)
codo = []
silueta = []
for k in range(1, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    codo.append((k, km.inertia_))
    if k >= 2:
        silueta.append((k, silhouette_score(X, km.labels_)))
km4 = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
datos["elegir_k"] = {"X": X, "c": km4.labels_, "codo": codo, "silueta": silueta}

# ---------------------------------------------------------------------------
# 5. Escalar las features: clientes con edad e ingreso
# ---------------------------------------------------------------------------
rng = np.random.default_rng(2)
n = 80
edad = np.r_[rng.normal(25, 3, n), rng.normal(55, 4, n)]
ingreso = rng.uniform(20000, 80000, 2 * n)  # en dólares al año: no separa a los grupos
Xc = np.c_[edad, ingreso]
sin_escalar = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(Xc)
escalado = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(Xc))
datos["escala"] = {"X": np.c_[edad, ingreso / 1000], "sin": sin_escalar, "con": escalado}

# ---------------------------------------------------------------------------
# 6. Colores de una foto
# ---------------------------------------------------------------------------
imagen = load_sample_image("china.jpg")
pixeles = imagen.reshape(-1, 3) / 255
muestra = pixeles[np.random.default_rng(0).choice(len(pixeles), 5000, replace=False)]
nube = muestra[:400]
plt.imsave(FIGS / "foto-original.png", imagen[::2, ::2])
paletas = {}
for k in [2, 4, 16]:
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(muestra)
    comprimida = km.cluster_centers_[km.predict(pixeles)].reshape(imagen.shape)
    plt.imsave(FIGS / f"foto-k{k}.png", np.clip(comprimida, 0, 1)[::2, ::2])
    paletas[k] = np.round(km.cluster_centers_ * 255).astype(int)
    if k == 4:
        nube_c = km.predict(nube)
datos["colores"] = {
    "paletas": paletas,
    "alto": imagen.shape[0],
    "ancho": imagen.shape[1],
    "nube": np.round(nube * 255).astype(int),
    "nube_c": nube_c,
}

# ---------------------------------------------------------------------------
# 7. Lunas: K-means falla, DBSCAN acierta
# ---------------------------------------------------------------------------
Xl, _ = make_moons(n_samples=300, noise=0.06, random_state=0)
atipicos = np.random.default_rng(5).uniform([-1.4, -0.9], [2.4, 1.4], size=(12, 2))
Xl = np.r_[Xl, atipicos]
km_lunas = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(Xl)
db_lunas = DBSCAN(eps=0.15, min_samples=5).fit_predict(Xl)
eps_valores = [0.05, 0.15, 0.5]
eps_paneles = [DBSCAN(eps=e, min_samples=5).fit_predict(Xl) for e in eps_valores]
vecinos = NearestNeighbors(n_neighbors=5).fit(Xl)
dist, _ = vecinos.kneighbors(Xl)
k_dist = np.sort(dist[:, -1])
datos["lunas"] = {
    "X": Xl,
    "kmeans": km_lunas,
    "dbscan": db_lunas,
    "eps": eps_valores,
    "eps_paneles": eps_paneles,
    "k_dist": [(i, d) for i, d in enumerate(k_dist)],
    "ruido": int((db_lunas == -1).sum()),
}
print("DBSCAN lunas: grupos", db_lunas.max() + 1, "ruido", (db_lunas == -1).sum())
for e, p in zip(eps_valores, eps_paneles):
    print(f"  eps={e}: grupos {p.max() + 1}, ruido {(p == -1).sum()}")

# ---------------------------------------------------------------------------
# 8. Tipos de puntos en DBSCAN (ejemplo pequeño)
# ---------------------------------------------------------------------------
P = np.array(
    [
        [1.0, 1.0], [1.6, 1.3], [1.2, 1.8], [1.9, 2.0], [2.6, 2.2], [2.3, 1.5],
        [3.4, 2.6], [5.0, 1.0], [4.8, 3.6],
    ]
)
EPS, MIN = 0.9, 3
vec = [np.flatnonzero(np.linalg.norm(P - p, axis=1) <= EPS) for p in P]
nucleo = np.array([len(v) >= MIN for v in vec])
tipo = []
for i, v in enumerate(vec):
    if nucleo[i]:
        tipo.append("nucleo")
    elif nucleo[v].any():
        tipo.append("frontera")
    else:
        tipo.append("ruido")
datos["tipos"] = {"P": P, "eps": EPS, "min": MIN, "tipo": tipo}
print("tipos:", tipo)

# ---------------------------------------------------------------------------
# 9. Densidades distintas
# ---------------------------------------------------------------------------
Xd = np.r_[
    make_blobs(n_samples=200, centers=[(0, 0)], cluster_std=0.3, random_state=0)[0],
    make_blobs(n_samples=60, centers=[(3, 0)], cluster_std=1.2, random_state=1)[0],
]
db_dens = DBSCAN(eps=0.3, min_samples=5).fit_predict(Xd)
datos["densidades"] = {"X": Xd, "c": db_dens}
print("densidades: grupos", db_dens.max() + 1, "ruido", (db_dens == -1).sum())

# ---------------------------------------------------------------------------
# 10. Dendrograma (ejemplo pequeño)
# ---------------------------------------------------------------------------
Q = np.array([[1.0, 1.0], [1.5, 1.6], [1.2, 2.4], [4.0, 1.2], [4.6, 1.8], [4.2, 3.6], [2.6, 4.6]])
nombres = list("ABCDEFG")
Z = linkage(Q, method="single")
dn = dendrogram(Z, no_plot=True, labels=nombres)
CORTE = 2.2
grupos_corte = fcluster(Z, t=CORTE, criterion="distance") - 1
datos["dendrograma"] = {
    "P": Q,
    "nombres": nombres,
    "icoord": dn["icoord"],
    "dcoord": dn["dcoord"],
    "hojas": dn["ivl"],
    "uniones": Z,
    "corte": CORTE,
    "grupos": grupos_corte,
}
print("dendrograma, alturas:", [round(z[2], 2) for z in Z], "hojas", dn["ivl"])

# ---------------------------------------------------------------------------
# 11. Enlace single contra Ward en las lunas, y la comparación final
# ---------------------------------------------------------------------------
Xm, _ = make_moons(n_samples=300, noise=0.06, random_state=0)
datos["enlace"] = {
    "X": Xm,
    "single": AgglomerativeClustering(n_clusters=2, linkage="single").fit_predict(Xm),
    "ward": AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(Xm),
}

conjuntos = {
    "bolas": (StandardScaler().fit_transform(make_blobs(n_samples=300, centers=3, cluster_std=1.3, random_state=8)[0]), 3, 0.35),
    "lunas": (StandardScaler().fit_transform(make_moons(n_samples=300, noise=0.06, random_state=0)[0]), 2, 0.3),
    "circulos": (StandardScaler().fit_transform(make_circles(n_samples=300, factor=0.45, noise=0.05, random_state=0)[0]), 2, 0.3),
}
comparacion = {}
for nombre, (Xs, k, eps) in conjuntos.items():
    comparacion[nombre] = {
        "X": Xs,
        "kmeans": KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xs),
        "dbscan": DBSCAN(eps=eps, min_samples=5).fit_predict(Xs),
        "single": AgglomerativeClustering(n_clusters=k, linkage="single").fit_predict(Xs),
    }
    for alg in ["kmeans", "dbscan", "single"]:
        c = comparacion[nombre][alg]
        print(f"  {nombre:9} {alg:7} grupos {c.max() + 1} ruido {(c == -1).sum()} tamaños {np.bincount(c[c >= 0])}")
datos["comparacion"] = comparacion

with open(FIGS / "datos.json", "w") as f:
    json.dump(limpiar(datos), f)
print("figs/09/datos.json")
