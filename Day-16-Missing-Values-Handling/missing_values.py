import pandas as pd
import matplotlib.pyplot as plt
import os

# Current folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create images folder
images_path = os.path.join(BASE_DIR, "images")
os.makedirs(images_path, exist_ok=True)

# Dataset path
dataset_path = os.path.join(BASE_DIR, "dataset.csv")

# Load dataset
data = pd.read_csv(dataset_path)

print("Original Dataset:")
print(data)

print("\nMissing Values:")
print(data.isnull().sum())

# Fill missing values using mean
filled_data = data.fillna(data.mean(numeric_only=True))

print("\nDataset After Handling Missing Values:")
print(filled_data)

# Visualization
before = data.isnull().sum()
after = filled_data.isnull().sum()

plt.figure(figsize=(8,5))

plt.bar(before.index, before.values, label="Before")

plt.bar(after.index, after.values)

plt.title("Missing Values Before and After Handling")

plt.xlabel("Columns")
plt.ylabel("Missing Values")

image_file = os.path.join(
    images_path,
    "missing_values_comparison.png"
)

plt.savefig(image_file)

plt.show()