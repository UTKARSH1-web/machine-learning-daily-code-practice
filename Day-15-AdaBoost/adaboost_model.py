from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
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

# Features and Target
X = data.drop("target", axis=1)
y = data["target"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# AdaBoost Model
model = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(
    model,
    os.path.join(
        models_path,
        "adaboost_model.pkl"
    )
)

# Feature Importance
importance = model.feature_importances_

plt.figure(figsize=(8,5))
plt.bar(X.columns, importance)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("AdaBoost Feature Importance")

plt.savefig(
    os.path.join(
        images_path,
        "adaboost_feature_importance.png"
    )
)

plt.show()