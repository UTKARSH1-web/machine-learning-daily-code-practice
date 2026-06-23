# Bagging Classifier

Bagging stands for Bootstrap Aggregating.

It trains multiple models on different random samples of the dataset and combines their predictions.

---

# Workflow

Original Dataset

↓

Bootstrap Sampling

↓

Train Multiple Models

↓

Aggregate Predictions

↓

Final Prediction

---

# Bootstrap Sampling

Sampling with replacement.

Example:

Original:

1 2 3 4 5

Sample:

1 2 2 5 5

---

# Why Bagging?

Reduces Variance

Improves Stability

Reduces Overfitting

---

# Formula

Final Prediction

=

Majority Vote

(Classification)

Average

(Regression)

---

# Advantages

- Better Accuracy
- Less Overfitting
- More Stable Models

---

# Disadvantages

- More Training Time
- More Memory Usage

---

# Real Example

Random Forest

=

Bagging

+

Random Feature Selection

---

# Applications

- Classification
- Regression
- Finance
- Healthcare

---

# Conclusion

Bagging combines multiple weak learners to create a stronger and more reliable model.