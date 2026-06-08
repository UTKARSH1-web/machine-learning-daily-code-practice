# AdaBoost (Adaptive Boosting)

AdaBoost is one of the earliest and most influential boosting algorithms.

It combines multiple weak learners to form a strong learner.

---

# How AdaBoost Works

Step 1:
Train a weak learner.

Step 2:
Identify wrongly classified samples.

Step 3:
Increase their importance (weight).

Step 4:
Train the next learner focusing more on those samples.

Step 5:
Combine all learners into a final model.

---

# Important Parameters

## n_estimators

Number of weak learners.

## learning_rate

Contribution of each learner.

---

# Advantages

- Easy to understand
- Improves weak learners
- Good accuracy
- Less parameter tuning

---

# Disadvantages

- Sensitive to noisy data
- Can overfit on outliers

---

# Applications

- Face Detection
- Customer Classification
- Fraud Detection
- Medical Diagnosis

---

# Conclusion

AdaBoost was the foundation of modern boosting algorithms such as Gradient Boosting, XGBoost, LightGBM, and CatBoost.