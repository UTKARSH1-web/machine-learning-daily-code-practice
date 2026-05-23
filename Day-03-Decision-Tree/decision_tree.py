from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os

# Base project folder
BASE_DIR = "Day-03-Decision-Tree"

# Create folders inside Day-03-Decision-Tree
models_path = os.path.join(BASE_DIR, "models")
images_path = os.path.join(BASE_DIR, "images")

os.makedirs(models_path, exist_ok=True)
os.makedirs(images_path, exist_ok=True)

# Load dataset
iris = load_iris()

# Features and target
X = iris.data
y = iris.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model inside Day-03-Decision-Tree/models
model_file = os.path.join(models_path, "decision_tree_model.pkl")

joblib.dump(model, model_file)

# Plot decision tree
plt.figure(figsize=(12, 8))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)

# Save image inside Day-03-Decision-Tree/images
image_file = os.path.join(images_path, "decision_tree.png")

plt.savefig(image_file)

# Show plot
plt.show()