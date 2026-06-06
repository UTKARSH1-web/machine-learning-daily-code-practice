from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os

plt.style.use('ggplot')

# Current folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create folders
models_path = os.path.join(BASE_DIR, "models")
images_path = os.path.join(BASE_DIR, "images")

os.makedirs(models_path, exist_ok=True)
os.makedirs(images_path, exist_ok=True)

# Dataset
dataset_path = os.path.join(BASE_DIR, "dataset.csv")
data = pd.read_csv(dataset_path)

# Features and Target
X = data.drop("target", axis=1)
y = data["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    min_child_samples=1,
    min_data_in_bin=1,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Save model
joblib.dump(
    model,
    os.path.join(
        models_path,
        "lightgbm_model.pkl"
    )
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
plt.figure(figsize=(6, 5))
disp.plot(cmap="Blues", colorbar=False)
plt.title("LightGBM Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(images_path, "lightgbm_confusion_matrix.png"), dpi=150)
plt.close()

# Feature Importance
importance = model.feature_importances_
colors = plt.cm.Set2(np.linspace(0, 1, len(X.columns)))
plt.figure(figsize=(8, 5))
bars = plt.bar(X.columns, importance, color=colors, edgecolor="black")
for bar, val in zip(bars, importance):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + (max(importance) * 0.02 if max(importance) > 0 else 0.02),
        f"{val:.0f}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="black"
    )

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("LightGBM Feature Importance")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(images_path, "lightgbm_feature_importance.png"), dpi=150)
plt.close()
