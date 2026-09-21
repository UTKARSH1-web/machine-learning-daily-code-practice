import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Current Folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Images Folder
images_path = os.path.join(BASE_DIR, "images")
os.makedirs(images_path, exist_ok=True)

# Dataset
dataset_path = os.path.join(BASE_DIR, "dataset.csv")

data = pd.read_csv(dataset_path)

# Features
X = data[["feature1", "feature2"]]

# Hierarchical Clustering
model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

clusters = model.fit_predict(X)

data["cluster"] = clusters

print(data)

# Cluster Plot
plt.figure(figsize=(8,6))

plt.scatter(
    data["feature1"],
    data["feature2"],
    c=data["cluster"],
    s=100
)

plt.title("Hierarchical Clustering")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.savefig(
    os.path.join(
        images_path,
        "hierarchical_clusters.png"
    )
)

plt.show()

# Dendrogram
linked = linkage(
    X,
    method="ward"
)

plt.figure(figsize=(10,6))

dendrogram(linked)

plt.title("Dendrogram")

plt.xlabel("Samples")

plt.ylabel("Distance")

plt.savefig(
    os.path.join(
        images_path,
        "dendrogram.png"
    )
)

plt.show()

#  # Kernel half-size: cover ±3 sigma
#         half = int(np.ceil(3 * sigma))

#         # Integer positions from -half to +half
#         x = np.arange(-half, half + 1)

#         # 1D Gaussian formula
#         g = (1 / (sigma * np.sqrt(2 * np.pi))) * \
#             np.exp(-(x ** 2) / (2 * sigma ** 2))

#         # Create 2D Gaussian using outer product
#         kernel = np.outer(g, g)

#         # Normalise so that kernel sums to 1
#         kernel = kernel / np.sum(kernel)

#         return kernel

# mode = kwargs.get("mode", "same")
# boundary = kwargs.get("boundary", "symm")
# fillvalue = kwargs.get("fillvalue", 0)

#         result = convolve2d(
#             image,
#             kernel,
#             mode=mode,
#             boundary=boundary,
#             fillvalue=fillvalue,
#         )

#         return result

# scale_factor = kwargs.get("scale_factor", 0.5)
# order = kwargs.get("order", 0)

#         result = rescale(
#             image,
#             scale_factor,
#             order=order,
#             anti_aliasing=False,
#             mode="reflect",
#         )

#         return result