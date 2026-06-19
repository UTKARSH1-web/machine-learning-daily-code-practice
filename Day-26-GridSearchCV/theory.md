# Hyperparameter Tuning

Hyperparameters are settings chosen before training.

Example:

Decision Tree

- max_depth
- criterion

Random Forest

- n_estimators
- max_depth

---

# Why Tune Hyperparameters?

Better performance

Better generalization

Higher accuracy

---

# Grid Search CV

Grid Search tries every possible parameter combination.

Example:

criterion:

- gini
- entropy

max_depth:

- 2
- 3
- 4

Combinations:

(gini,2)

(gini,3)

(gini,4)

(entropy,2)

(entropy,3)

(entropy,4)

---

Each combination is evaluated using Cross Validation.

Best combination is selected.

---

# Advantages

- Finds best parameters
- Easy to use
- Reliable

---

# Disadvantages

- Computationally expensive
- Slow for large grids

---

# Applications

- Model Optimization
- Production ML
- Competitions
- Research

---

# Conclusion

Grid Search CV automates hyperparameter tuning and helps find the best model configuration.