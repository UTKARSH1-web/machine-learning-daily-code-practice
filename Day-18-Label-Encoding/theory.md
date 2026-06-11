# Label Encoding

Label Encoding converts categorical text values into numerical values.

---

# Example

Before:

Male
Female
Male

After:

1
0
1

---

# Why Label Encoding?

Machine Learning algorithms cannot process text directly.

They require numerical input.

---

# LabelEncoder

Scikit-learn provides:

from sklearn.preprocessing import LabelEncoder

---

# Example

Colors:

Red
Blue
Green

Encoded:

Blue  -> 0
Green -> 1
Red   -> 2

---

# Advantages

- Simple
- Fast
- Easy implementation

---

# Disadvantages

Creates artificial order.

Example:

Red = 2
Blue = 0

Model may think Red > Blue.

---

# When to Use?

Good for:

- Binary categories
- Target labels

Examples:

Yes / No
Pass / Fail
Male / Female

---

# When NOT to Use?

For multiple categories like:

Delhi
Mumbai
Chennai

Use One-Hot Encoding instead.

---

# Conclusion

Label Encoding is useful for converting categorical values into numbers for machine learning models.