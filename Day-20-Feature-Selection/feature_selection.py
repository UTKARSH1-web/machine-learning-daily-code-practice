import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Current Folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create folders
models_path = os.path.join(BASE_DIR, "models")
images_path = os.path.join(BASE_DIR, "images")

os.makedirs(models_path, exist_ok=True)
os.makedirs(images_path, exist_ok=True)

# Dataset
dataset_path = os.path.join(BASE_DIR, "dataset.csv")

data = pd.read_csv(dataset_path)

print("Original Dataset:")
print(data.head())

# Features and Target
X = data.drop("target", axis=1)
y = data["target"]

# Select Top 3 Features
selector = SelectKBest(
    score_func=f_classif,
    k=3
)

X_selected = selector.fit_transform(X, y)

# Scores
scores = pd.DataFrame({
    "Feature": X.columns,
    "Score": selector.scores_
})

scores = scores.sort_values(
    by="Score",
    ascending=False
)

print("\nFeature Scores:")
print(scores)

# Save Selector
joblib.dump(
    selector,
    os.path.join(
        models_path,
        "feature_selector.pkl"
    )
)

# Graph
plt.figure(figsize=(8,5))

plt.bar(
    scores["Feature"],
    scores["Score"]
)

plt.title("Feature Importance Scores")
plt.xlabel("Features")
plt.ylabel("Score")

plt.savefig(
    os.path.join(
        images_path,
        "feature_importance.png"
    )
)

plt.show()

print("\nSelected Features:")

selected_columns = X.columns[
    selector.get_support()
]

print(selected_columns)