# Outlier Detection

Outliers are unusual data points that differ significantly from other observations.

---

# Why Outliers Matter?

Problems:

- Reduce model accuracy
- Distort statistics
- Cause misleading results

---

# Methods for Detecting Outliers

## 1. IQR Method

IQR = Q3 - Q1

Where:

Q1 = 25th Percentile

Q3 = 75th Percentile

---

Lower Bound

```text
Q1 - 1.5 × IQR
```

Upper Bound

```text
Q3 + 1.5 × IQR
```

Values outside these limits are outliers.

---

## 2. Z-Score Method

Formula:

```text
Z = (X - Mean) / Standard Deviation
```

Rule:

|Z| > 3

Then value is considered an outlier.

---

# Advantages

- Better data quality
- Improved model accuracy
- More reliable statistics

---

# Disadvantages

- Real important values may be removed
- Not all unusual values are wrong

---

# Applications

- Fraud Detection
- Healthcare
- Finance
- Data Cleaning

---

# Conclusion

Outlier Detection helps improve machine learning model performance by removing extreme values.