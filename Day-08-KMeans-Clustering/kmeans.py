from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os

# Current folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create folders
models_path = os.path.join(BASE_DIR, "models")
images_path = os.path.join(BASE_DIR, "images")

os.makedirs(models_path, exist_ok=True)
os.makedirs(images_path, exist_ok=True)

# Dataset path
dataset_path = os.path.join(BASE_DIR, "dataset.csv")

# Load dataset
data = pd.read_csv(dataset_path)

# Features
X = data[["x", "y"]]

# Create KMeans model
model = KMeans(
    n_clusters=4,
    random_state=42
)

# Train model
model.fit(X)

# Cluster labels
labels = model.labels_

# Cluster centers
centers = model.cluster_centers_

# Save model
model_file = os.path.join(
    models_path,
    "kmeans_model.pkl"
)

joblib.dump(model, model_file)

# Plot clusters
plt.figure(figsize=(8,6))

plt.scatter(
    data["x"],
    data["y"],
    c=labels
)

# Plot centroids
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker='X',
    s=200
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-Means Clustering")

# Save graph
image_file = os.path.join(
    images_path,
    "kmeans_clusters.png"
)

plt.savefig(image_file)

plt.show()