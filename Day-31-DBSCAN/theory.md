# DBSCAN

Density-Based Spatial Clustering of Applications with Noise

---

# Idea

Points with high density form clusters.

Points with low density become noise.

---

# Parameters

## eps

Neighborhood radius.

---

## min_samples

Minimum points required to form a cluster.

---

# Output

Cluster Labels:

0

1

2

...

Noise:

-1

---

# Advantages

- No need to specify number of clusters
- Detects outliers
- Handles irregular clusters

---

# Disadvantages

- Sensitive to eps
- Struggles in high dimensions

---

# Applications

- Customer Segmentation
- Fraud Detection
- GPS Data Analysis
- Anomaly Detection

---

# Conclusion

DBSCAN creates clusters based on density and identifies noise points automatically.