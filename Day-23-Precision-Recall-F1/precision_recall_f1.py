import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# Current Folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create Images Folder
images_path = os.path.join(BASE_DIR, "images")
os.makedirs(images_path, exist_ok=True)

# Dataset Path
dataset_path = os.path.join(BASE_DIR, "dataset.csv")

# Load Dataset
data = pd.read_csv(dataset_path)

# Features & Target
X = data.drop("target", axis=1)
y = data["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Model
model = DecisionTreeClassifier(
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

print(f"Accuracy : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1 Score : {f1:.2f}")

# Graph
metrics = [
    accuracy,
    precision,
    recall,
    f1
]

names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1"
]

plt.figure(figsize=(8,5))

plt.bar(
    names,
    metrics
)

plt.ylim(0, 1)

plt.title(
    "Evaluation Metrics Comparison"
)

plt.savefig(
    os.path.join(
        images_path,
        "metrics_comparison.png"
    )
)

plt.show()