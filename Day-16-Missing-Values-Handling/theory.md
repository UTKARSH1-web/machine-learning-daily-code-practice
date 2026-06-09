# Missing Values Handling

Missing values occur when data is incomplete.

Most machine learning algorithms cannot handle missing values directly.

---

# Why Missing Values Matter?

- Reduce model accuracy
- Create bias
- Cause training errors

---

# Common Methods

## Mean Imputation

Replace missing values with column mean.

Example:

Age Column

21
22
23
NaN

Mean = 22

Replace NaN with 22

---

## Median Imputation

Replace with median value.

Useful for outliers.

---

## Mode Imputation

Replace with most frequent value.

Used for categorical data.

---

## Delete Rows

Remove rows containing missing values.

Use only when missing data is small.

---

# Pandas Functions

Check missing values:

data.isnull().sum()

Fill with mean:

data.fillna(data.mean())

Fill with median:

data.fillna(data.median())

Drop rows:

data.dropna()

---

# Advantages

- Improves data quality
- Prevents training errors
- Better model performance

---

# Disadvantages

- Wrong imputation can reduce accuracy
- Data information may be lost

---

# Conclusion

Handling missing values is one of the most important steps in data preprocessing.