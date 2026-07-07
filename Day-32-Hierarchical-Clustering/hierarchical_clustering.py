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