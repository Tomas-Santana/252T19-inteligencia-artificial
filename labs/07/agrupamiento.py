"""Clase 9, laboratorio: agrupamiento.

Notación:
  - X: matriz de forma (n, d), un ejemplo x^(i) por fila.
  - centroids: matriz de forma (k, d), un centroide mu_j por fila.
  - c: vector de forma (n,) con el grupo de cada ejemplo, c^(i) en {0, ..., k-1}.
    (En Python los grupos se numeran desde 0.)

Recordatorios de NumPy:
  - X[c == j] son las filas de X cuyo grupo es j.
  - A.sum(axis=1) suma cada fila; A.mean(axis=0) promedia cada columna.
  - np.argmin(A, axis=1) devuelve, para cada fila, la posición del valor más pequeño.
"""

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

# ---------------------------------------------------------------------------
# Parte 1: K-means
# ---------------------------------------------------------------------------


def assign_clusters(X, centroids):
    """Paso de asignación: c^(i) = argmin_j ||x^(i) - mu_j||^2.

    Devuelve un vector de enteros de forma (n,).
    """
    n, k = len(X), len(centroids)
    distances = np.zeros((n, k))  # distances[i, j] = ||x^(i) - mu_j||^2
    # CÓDIGO AQUÍ:
    #   1. para cada centroide j, llenar la columna distances[:, j] con la
    #      distancia al cuadrado de cada ejemplo a centroids[j]
    #   2. devolver, para cada ejemplo, la posición de la columna más pequeña
    raise NotImplementedError("Falta completar assign_clusters en agrupamiento.py")


def update_centroids(X, c, k):
    """Paso de mover: mu_j = promedio de los ejemplos del grupo j.

    Devuelve una matriz de forma (k, d). Se puede suponer que ningún grupo está vacío.
    """
    centroids = np.zeros((k, X.shape[1]))
    # CÓDIGO AQUÍ: para cada grupo j, centroids[j] es el promedio de las filas de X del grupo j
    raise NotImplementedError("Falta completar update_centroids en agrupamiento.py")


def distortion(X, c, centroids):
    """La distorsión J(c, mu) = sum_i ||x^(i) - mu_c^(i)||^2.

    Devuelve un número (float).
    """
    # CÓDIGO AQUÍ. Pista: centroids[c] es una matriz (n, d) con el centroide de cada ejemplo.
    raise NotImplementedError("Falta completar distortion en agrupamiento.py")


# ---------------------------------------------------------------------------
# Parte 2: estandarizar
# ---------------------------------------------------------------------------


def standardize(X):
    """Resta a cada columna (feature) su media y la divide entre su desviación estándar.

    Devuelve una matriz de la misma forma que X, con columnas de media 0 y desviación 1.
    """
    # CÓDIGO AQUÍ. Pista: X.mean(axis=0) y X.std(axis=0) dan un número por columna.
    raise NotImplementedError("Falta completar standardize en agrupamiento.py")


# ---------------------------------------------------------------------------
# Parte 3: los tres algoritmos de scikit-learn
# ---------------------------------------------------------------------------


def compare_algorithms(X, k, eps, min_samples=5):
    """Agrupa X con los tres métodos de las notas.

    Devuelve un diccionario con el grupo de cada ejemplo según cada método:
        {"kmeans": ..., "dbscan": ..., "single": ...}
    Cada valor es un vector de forma (n,). En DBSCAN, el ruido es -1.
    """
    # CÓDIGO AQUÍ. Crear cada modelo y llamar a fit_predict(X):
    #   "kmeans": KMeans con n_clusters=k, n_init=10, random_state=0
    #   "dbscan": DBSCAN con eps y min_samples
    #   "single": AgglomerativeClustering con n_clusters=k y linkage="single"
    raise NotImplementedError("Falta completar compare_algorithms en agrupamiento.py")
