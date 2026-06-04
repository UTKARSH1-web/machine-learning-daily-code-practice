from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
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

# Gradient Boosting Model
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Save Model
model_file = os.path.join(
    models_path,
    "gradient_boosting_model.pkl"
)

joblib.dump(model, model_file)

# Accuracy Graph
plt.figure(figsize=(6,5))

plt.bar(
    ["Gradient Boosting"],
    [accuracy * 100]
)

plt.ylabel("Accuracy (%)")
plt.title("Gradient Boosting Accuracy")

# Save Graph
image_file = os.path.join(
    images_path,
    "gradient_boosting_accuracy.png"
)

plt.savefig(image_file)

plt.show()