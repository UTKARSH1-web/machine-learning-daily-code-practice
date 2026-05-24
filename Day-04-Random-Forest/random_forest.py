from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os

# Base project folder (script directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create folders
models_path = os.path.join(BASE_DIR, "models")
images_path = os.path.join(BASE_DIR, "images")

dataset_path = os.path.join(BASE_DIR, "dataset.csv")

os.makedirs(models_path, exist_ok=True)
os.makedirs(images_path, exist_ok=True)

# Load dataset
data = pd.read_csv(dataset_path)

# Features and target
X = data.drop("target", axis=1)
y = data["target"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model
model_file = os.path.join(
    models_path,
    "random_forest_model.pkl"
)

joblib.dump(model, model_file)

# Feature importance
importance = model.feature_importances_

feature_names = X.columns

# Plot graph
plt.figure(figsize=(8,5))

plt.bar(feature_names, importance)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")

# Save graph
image_file = os.path.join(
    images_path,
    "random_forest_importance.png"
)

plt.savefig(image_file)

plt.show()