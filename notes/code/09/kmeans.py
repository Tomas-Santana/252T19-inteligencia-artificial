from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=3, random_state=4)  # sin usar las etiquetas

kmeans = KMeans(n_clusters=3, n_init=10, random_state=0)
labels = kmeans.fit_predict(X)            # un grupo por ejemplo

print(labels[:10])
print(kmeans.cluster_centers_.round(2))   # un centroide por fila
print(round(kmeans.inertia_, 1))          # la distorsión J
print(kmeans.predict([[4, -5], [9, 4]]))  # grupos de ejemplos nuevos
