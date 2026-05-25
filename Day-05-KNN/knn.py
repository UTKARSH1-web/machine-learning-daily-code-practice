from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
model = KNeighborsClassifier(
    n_neighbors=3
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
    "knn_model.pkl"
)

joblib.dump(model, model_file)

# Accuracy graph
plt.figure(figsize=(6,5))

plt.bar(["KNN"], [accuracy * 100])

plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy")

# Save graph
image_file = os.path.join(
    images_path,
    "knn_accuracy.png"
)

plt.savefig(image_file)

plt.show()