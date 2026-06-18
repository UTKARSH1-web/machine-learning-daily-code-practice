# Cross Validation

Cross Validation is a technique used to evaluate machine learning models more reliably.

---

# Why Cross Validation?

Train-Test Split results depend on data split.

Cross Validation reduces this dependency.

---

# K-Fold Cross Validation

Dataset is divided into K equal parts.

Each fold gets a chance to become the test set.

---

# Example

K = 5

Fold1
Fold2
Fold3
Fold4
Fold5

---

Iteration 1

Test = Fold1

Train = Remaining

---

Iteration 2

Test = Fold2

Train = Remaining

...

---

Final Score

Average of all folds.

---

# Advantages

- Better evaluation
- Less variance
- More reliable results

---

# Disadvantages

- More computation
- Slower than train-test split

---

# Common Values

K = 5

K = 10

Most common in industry.

---

# Applications

- Model Selection
- Hyperparameter Tuning
- Research
- Production ML

---

# Conclusion

Cross Validation provides a more trustworthy estimate of model performance.