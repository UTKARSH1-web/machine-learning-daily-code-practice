import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import VotingClassifier

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

# Individual Models
dt = DecisionTreeClassifier(
    random_state=42
)

knn = KNeighborsClassifier(
    n_neighbors=3
)

nb = GaussianNB()

# Voting Classifier
voting_model = VotingClassifier(
    estimators=[
        ("dt", dt),
        ("knn", knn),
        ("nb", nb)
    ],
    voting="hard"
)

# Train
voting_model.fit(
    X_train,
    y_train
)

# Predict
y_pred = voting_model.predict(
    X_test
)

# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"Voting Accuracy: {accuracy:.2f}"
)

# Save Model
joblib.dump(
    voting_model,
    os.path.join(
        models_path,
        "voting_model.pkl"
    )
)

# Compare Models
models = {
    "Decision Tree": dt,
    "KNN": knn,
    "Naive Bayes": nb,
    "Voting": voting_model
}

scores = []

for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    score = accuracy_score(
        y_test,
        pred
    )

    scores.append(score)

# Plot
plt.figure(figsize=(8,5))

plt.bar(
    models.keys(),
    scores
)

plt.ylim(0,1)

plt.title(
    "Model Accuracy Comparison"
)

plt.xticks(rotation=20)

plt.savefig(
    os.path.join(
        images_path,
        "model_accuracy_comparison.png"
    )
)

plt.show()