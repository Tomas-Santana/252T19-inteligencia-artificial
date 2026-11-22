from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.9, random_state=7)

for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = kmeans.fit_predict(X)
    print(f"k={k}  J={kmeans.inertia_:8.1f}  silueta={silhouette_score(X, labels):.3f}")
