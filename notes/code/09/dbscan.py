import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=300, noise=0.06, random_state=0)

dbscan = DBSCAN(eps=0.15, min_samples=5)
labels = dbscan.fit_predict(X)               # -1 = ruido

print(np.unique(labels))
print("grupos:", labels.max() + 1)
print("ruido:", (labels == -1).sum(), "puntos")
