import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import LabelEncoder

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

# Label Encoder
encoder = LabelEncoder()

# Encode columns
data["gender"] = encoder.fit_transform(data["gender"])

data["city"] = encoder.fit_transform(data["city"])

data["purchased"] = encoder.fit_transform(data["purchased"])

print("\nEncoded Dataset:\n")
print(data)

# Visualization
plt.figure(figsize=(6,4))

data["city"].value_counts().plot(kind="bar")

plt.title("Encoded City Distribution")

image_file = os.path.join(
    images_path,
    "label_encoding_comparison.png"
)

plt.savefig(image_file)

plt.show()