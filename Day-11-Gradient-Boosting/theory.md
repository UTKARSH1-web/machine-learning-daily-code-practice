# Gradient Boosting

Gradient Boosting is an ensemble learning algorithm.

It combines multiple weak learners (usually Decision Trees)
to create a strong predictive model.

---

# How It Works

Step 1:
Train first Decision Tree

Step 2:
Calculate errors

Step 3:
Train next tree on previous errors

Step 4:
Repeat until model improves

---

# Important Parameters

n_estimators
Number of trees

learning_rate
Controls contribution of each tree

max_depth
Controls tree depth

---

# Advantages

- High accuracy
- Handles complex datasets
- Reduces bias

---

# Disadvantages

- Slower training
- Sensitive to overfitting

---

# Applications

- Fraud Detection
- Customer Churn Prediction
- Recommendation Systems
- Ranking Systems

---

# Conclusion

Gradient Boosting is one of the most powerful machine learning algorithms and forms the foundation of XGBoost, LightGBM, and CatBoost.