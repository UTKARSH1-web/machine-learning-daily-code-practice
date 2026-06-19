import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

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

# Model
model = DecisionTreeClassifier(
    random_state=42
)

# Hyperparameter Grid
param_grid = {
    "criterion": [
        "gini",
        "entropy"
    ],
    "max_depth": [
        2,
        3,
        4,
        5
    ]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X, y)

# Results
print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Score:")
print(grid_search.best_score_)

# Save Best Model
joblib.dump(
    grid_search.best_estimator_,
    os.path.join(
        models_path,
        "best_model.pkl"
    )
)

# Visualization
results = pd.DataFrame(
    grid_search.cv_results_
)

plt.figure(figsize=(8,5))

plt.plot(
    results["mean_test_score"]
)

plt.title(
    "Grid Search Scores"
)

plt.xlabel(
    "Parameter Combination"
)

plt.ylabel(
    "Accuracy"
)

plt.savefig(
    os.path.join(
        images_path,
        "grid_search_scores.png"
    )
)

plt.show()