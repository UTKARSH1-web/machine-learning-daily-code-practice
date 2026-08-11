import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    roc_curve,
    auc
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

# Train Test Split
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

# Probability Prediction
y_prob = model.predict_proba(X_test)[:, 1]

# ROC Values
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

# AUC Score
roc_auc = auc(
    fpr,
    tpr
)

print(f"AUC Score: {roc_auc:.2f}")

# Plot ROC Curve
plt.figure(figsize=(8,5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.2f}"
)

# Random Classifier Line
plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig(
    os.path.join(
        images_path,
        "roc_curve.png"
    )
)

plt.show()