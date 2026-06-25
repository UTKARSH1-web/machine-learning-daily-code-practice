import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import DBSCAN

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

# DBSCAN Model
dbscan = DBSCAN(
    eps=1.0,
    min_samples=3
)

# Train
clusters = dbscan.fit_predict(X)

# Add Cluster Labels
data["cluster"] = clusters

print(data)

# Visualization
plt.figure(figsize=(8,6))

scatter = plt.scatter(
    data["feature1"],
    data["feature2"],
    c=data["cluster"]
)

plt.title(
    "DBSCAN Clustering"
)

plt.xlabel(
    "Feature 1"
)

plt.ylabel(
    "Feature 2"
)

plt.savefig(
    os.path.join(
        images_path,
        "dbscan_clusters.png"
    )
)

plt.show()