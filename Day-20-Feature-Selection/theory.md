# Feature Selection

Feature Selection is the process of choosing the most important features for machine learning.

---

# Why Feature Selection?

Benefits:

- Faster Training
- Better Accuracy
- Less Overfitting
- Simpler Models

---

# Types of Feature Selection

## Filter Methods

Example:

- Correlation
- ANOVA
- Chi-Square

---

## Wrapper Methods

Example:

- Forward Selection
- Backward Elimination

---

## Embedded Methods

Example:

- Random Forest Feature Importance
- Lasso Regression

---

# SelectKBest

Scikit-learn provides:

SelectKBest()

It selects top K features based on score.

---

# Example

Features:

Study Hours
Attendance
Assignments
Social Media Hours

Scores:

Study Hours = 95

Attendance = 80

Assignments = 70

Social Media = 10

Selected:

Study Hours
Attendance
Assignments

---

# Advantages

- Faster Models
- Better Performance
- Removes Noise

---

# Disadvantages

- Important features may sometimes be removed
- Depends on selection method

---

# Conclusion

Feature Selection helps build efficient and accurate machine learning models.