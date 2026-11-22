from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=300, noise=0.06, random_state=0)

for linkage in ["single", "ward"]:
    model = AgglomerativeClustering(n_clusters=2, linkage=linkage)
    labels = model.fit_predict(X)
    print(linkage, "-> tamaños de los grupos:", (labels == 0).sum(), (labels == 1).sum())
