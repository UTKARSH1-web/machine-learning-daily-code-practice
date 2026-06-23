import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

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

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Single Tree
tree = DecisionTreeClassifier(
    random_state=42
)

tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)

tree_acc = accuracy_score(
    y_test,
    tree_pred
)

# Bagging Model
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=42
)

bagging.fit(X_train, y_train)

bag_pred = bagging.predict(X_test)

bag_acc = accuracy_score(
    y_test,
    bag_pred
)

print(f"Decision Tree Accuracy : {tree_acc:.2f}")
print(f"Bagging Accuracy       : {bag_acc:.2f}")

# Save Model
joblib.dump(
    bagging,
    os.path.join(
        models_path,
        "bagging_model.pkl"
    )
)

# Graph
plt.figure(figsize=(6,5))

plt.bar(
    ["Decision Tree", "Bagging"],
    [tree_acc, bag_acc]
)

plt.ylim(0,1)

plt.title(
    "Bagging vs Decision Tree"
)

plt.savefig(
    os.path.join(
        images_path,
        "bagging_vs_tree.png"
    )
)

plt.show()