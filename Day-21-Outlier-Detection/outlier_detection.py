import pandas as pd
import matplotlib.pyplot as plt
import os

# Current Folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create Images Folder
images_path = os.path.join(BASE_DIR, "images")
os.makedirs(images_path, exist_ok=True)

# Load Dataset
dataset_path = os.path.join(BASE_DIR, "dataset.csv")

data = pd.read_csv(dataset_path)

print("Original Dataset:")
print(data)

# Boxplot Before
plt.figure(figsize=(8,5))

plt.boxplot(data["salary"])

plt.title("Before Removing Outliers")

plt.savefig(
    os.path.join(
        images_path,
        "boxplot_before.png"
    )
)

plt.show()

# IQR Method
Q1 = data["salary"].quantile(0.25)
Q3 = data["salary"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)

print("\nQ1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)

print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

# Remove Outliers
clean_data = data[
    (data["salary"] >= lower_bound)
    &
    (data["salary"] <= upper_bound)
]

print("\nDataset After Removing Outliers:")
print(clean_data)

# Boxplot After
plt.figure(figsize=(8,5))

plt.boxplot(clean_data["salary"])

plt.title("After Removing Outliers")

plt.savefig(
    os.path.join(
        images_path,
        "boxplot_after.png"
    )
)

plt.show()