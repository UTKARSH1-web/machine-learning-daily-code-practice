import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Current Folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folders
models_path = os.path.join(BASE_DIR, "models")
images_path = os.path.join(BASE_DIR, "images")

os.makedirs(models_path, exist_ok=True)
os.makedirs(images_path, exist_ok=True)

# Dataset
dataset_path = os.path.join(BASE_DIR, "dataset.csv")

data = pd.read_csv(dataset_path)

# Features & Target
X = data.drop("target", axis=1)
y = data["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=3))
])

# Train
pipeline.fit(X_train, y_train)

# Prediction
y_pred = pipeline.predict(X_test)

# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"Accuracy: {accuracy:.2f}")

# Save Pipeline
joblib.dump(
    pipeline,
    os.path.join(
        models_path,
        "pipeline_model.pkl"
    )
)

# Visualization
plt.figure(figsize=(5,4))

plt.bar(
    ["Accuracy"],
    [accuracy]
)

plt.ylim(0,1)

plt.title(
    "Pipeline Model Accuracy"
)

plt.savefig(
    os.path.join(
        images_path,
        "pipeline_accuracy.png"
    )
)

plt.show()