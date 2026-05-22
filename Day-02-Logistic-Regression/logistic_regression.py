import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset (resolve path relative to this script)
csv_path = Path(__file__).resolve().parent / "dataset.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"Dataset not found at {csv_path}.\nMake sure 'dataset.csv' is present next to this script or run the script from the Day-02-Logistic-Regression folder.")

df = pd.read_csv(csv_path)

print(df.head())

# Features and Labels
X = df[["Hours"]]
y = df["Pass"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = LogisticRegression()

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Predictions:", y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Visualization
plt.scatter(df["Hours"], df["Pass"])

plt.xlabel("Study Hours")
plt.ylabel("Pass/Fail")

plt.title("Logistic Regression")

plt.show()