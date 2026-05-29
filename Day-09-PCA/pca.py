from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os

#current folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#Create Folders
models_path = os.path.join(BASE_DIR, "models")
images_path = os.path.join(BASE_DIR, "images")

os.makedirs(models_path, exist_ok=True)
os.makedirs(images_path, exist_ok=True)

#Dataset path
dataset_path = os.path.join(BASE_DIR, "dataset.csv")

#Load Dataset
data = pd.read_csv(dataset_path)

#Features
X = data

# Create PCA model
model = PCA(n_components=2)

# Transform data
X_pca = model.fit_transform(X)

# Save model
model_file = os.path.join(
    models_path,
    "pca_model.pkl"
)

joblib.dump(model, model_file)

#Create visualization
plt.figure(figsize=(8, 6))

plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization")

# Save graph
image_file = os.path.join(images_path, "pca_visualization.png")
plt.savefig(image_file)
plt.show()