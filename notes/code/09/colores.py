import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

image = load_sample_image("china.jpg")       # (alto, ancho, 3): RGB de 0 a 255
pixels = image.reshape(-1, 3) / 255          # un ejemplo por píxel
print(image.shape, "->", pixels.shape)

rng = np.random.default_rng(0)
sample = pixels[rng.choice(len(pixels), 5000, replace=False)]
kmeans = KMeans(n_clusters=16, n_init=4, random_state=0).fit(sample)

labels = kmeans.predict(pixels)              # el color más cercano a cada píxel
compressed = kmeans.cluster_centers_[labels].reshape(image.shape)
print(kmeans.cluster_centers_.shape, "colores")
