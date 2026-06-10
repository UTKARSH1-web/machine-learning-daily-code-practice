import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Current folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create images folder
images_path = os.path.join(BASE_DIR, "images")
os.makedirs(images_path, exist_ok=True)

# Dataset path
dataset_path = os.path.join(BASE_DIR, "dataset.csv")

# Load dataset
data = pd.read_csv(dataset_path)

print("Original Dataset:\n")
print(data)

# Features
X = data.copy()

# Standard Scaling
standard_scaler = StandardScaler()

X_standard = standard_scaler.fit_transform(X)

standard_df = pd.DataFrame(
    X_standard,
    columns=X.columns
)

print("\nStandard Scaled Data:\n")
print(standard_df)

# MinMax Scaling
minmax_scaler = MinMaxScaler()

X_minmax = minmax_scaler.fit_transform(X)

minmax_df = pd.DataFrame(
    X_minmax,
    columns=X.columns
)

print("\nMinMax Scaled Data:\n")
print(minmax_df)

# Visualization
plt.figure(figsize=(8,5))

plt.plot(
    data["salary"],
    label="Original Salary"
)

plt.plot(
    minmax_df["salary"],
    label="Scaled Salary"
)

plt.title("Feature Scaling Comparison")
plt.legend()

image_file = os.path.join(
    images_path,
    "scaling_comparison.png"
)

plt.savefig(image_file)

plt.show()