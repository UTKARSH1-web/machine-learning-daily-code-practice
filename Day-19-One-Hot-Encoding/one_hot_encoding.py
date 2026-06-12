import pandas as pd
import matplotlib.pyplot as plt
import os

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

# One Hot Encoding
encoded_data = pd.get_dummies(
    data,
    columns=["gender", "city"],
    dtype=int
)

print("\nOne Hot Encoded Dataset:\n")
print(encoded_data)

# Visualization
city_counts = data["city"].value_counts()

plt.figure(figsize=(6,4))

city_counts.plot(kind="bar")

plt.title("City Distribution")

image_file = os.path.join(
    images_path,
    "one_hot_encoding_distribution.png"
)

plt.savefig(image_file)

plt.show()