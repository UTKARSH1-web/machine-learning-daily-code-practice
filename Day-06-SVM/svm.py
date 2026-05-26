from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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

# Create SVM model
model = SVC(kernel='linear')

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
    "svm_model.pkl"
)

joblib.dump(model, model_file)

# Scatter Plot Visualization
plt.figure(figsize=(7,6))

# Plot data points
plt.scatter(
    data["feature1"],
    data["feature2"],
    c=data["target"]
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM Dataset Visualization")

# Save graph
image_file = os.path.join(
    images_path,
    "svm_accuracy.png"
)

plt.savefig(image_file)

plt.show()