import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Current Folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create Images Folder
images_path = os.path.join(BASE_DIR, "images")
os.makedirs(images_path, exist_ok=True)

# Dataset
dataset_path = os.path.join(BASE_DIR, "dataset.csv")

data = pd.read_csv(dataset_path)

# Features and Target
X = data.drop("target", axis=1)
y = data["target"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Model
model = DecisionTreeClassifier(
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred
)

print("Confusion Matrix:")
print(cm)

# Plot
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Confusion Matrix")

plt.savefig(
    os.path.join(
        images_path,
        "confusion_matrix.png"
    )
)

plt.show()