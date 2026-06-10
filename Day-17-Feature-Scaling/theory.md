# Feature Scaling

Feature Scaling is a preprocessing technique used to bring all features to a similar scale.

---

# Why Feature Scaling?

Example:

Age = 25

Salary = 500000

Experience = 5

Salary dominates because its values are much larger.

---

# Types of Scaling

## StandardScaler

Formula:

z = (x - mean) / standard deviation

Result:

- Mean = 0
- Standard Deviation = 1

Used in:

- Logistic Regression
- SVM
- Neural Networks
- PCA

---

## MinMaxScaler

Formula:

(x - min) / (max - min)

Range:

0 to 1

Used in:

- Neural Networks
- KNN
- Deep Learning

---

# Advantages

- Faster training
- Better accuracy
- Prevents feature domination

---

# Algorithms That Need Scaling

✔ KNN

✔ SVM

✔ Logistic Regression

✔ Neural Networks

✔ PCA

---

# Algorithms That Usually Don't Need Scaling

✔ Decision Tree

✔ Random Forest

✔ XGBoost

✔ LightGBM

✔ CatBoost

---

# Conclusion

Feature Scaling improves model performance when features have different ranges.