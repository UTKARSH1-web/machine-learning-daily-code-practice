import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Current Folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Images Folder
images_path = os.path.join(BASE_DIR, "images")
os.makedirs(images_path, exist_ok=True)

# Dataset
dataset_path = os.path.join(BASE_DIR, "dataset.csv")

data = pd.read_csv(dataset_path)

# Features & Target
X = data.drop("target", axis=1)
y = data["target"]

# Model
model = DecisionTreeClassifier(
    random_state=42
)

# KFold
kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Cross Validation
scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold,
    scoring="accuracy"
)

print("Fold Scores:")

for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: {score:.2f}")

print("\nAverage Accuracy:")
print(scores.mean())

# Graph
plt.figure(figsize=(8,5))

plt.bar(
    range(1,6),
    scores
)

plt.xlabel("Fold")
plt.ylabel("Accuracy")

plt.title("K-Fold Cross Validation Scores")

plt.ylim(0,1)

plt.savefig(
    os.path.join(
        images_path,
        "kfold_scores.png"
    )
)

plt.show()