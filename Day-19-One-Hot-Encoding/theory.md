# One-Hot Encoding

One-Hot Encoding converts categorical values into separate binary columns.

---

# Example

Original:

City

Delhi
Mumbai
Chennai

---

After Encoding

Delhi Mumbai Chennai

1      0      0
0      1      0
0      0      1

---

# Why Use One-Hot Encoding?

Label Encoding creates artificial order.

Example:

Delhi = 1
Mumbai = 2

Model may think Mumbai > Delhi.

One-Hot Encoding removes this issue.

---

# Pandas Function

pd.get_dummies()

Example:

pd.get_dummies(data)

---

# Advantages

- No artificial ordering
- Better for categorical features
- Widely used in ML

---

# Disadvantages

- Creates many columns
- Higher memory usage

---

# When to Use?

Good for:

- City
- Country
- Department
- Product Category

---

# When Not to Use?

Very high-cardinality columns.

Example:

10000 unique customer IDs.

---

# Conclusion

One-Hot Encoding is one of the most important preprocessing techniques for categorical data.