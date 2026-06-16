# Precision, Recall & F1 Score

These metrics are used to evaluate classification models.

They are derived from the Confusion Matrix.

---

# Confusion Matrix

             Predicted

           No      Yes

Actual No  TN      FP

Actual Yes FN      TP

---

# Precision

Formula:

TP / (TP + FP)

Question:

Out of all predicted positives,
how many were actually positive?

---

Example

TP = 80

FP = 20

Precision:

80 / (80 + 20)

= 0.80

= 80%

---

# Recall

Formula:

TP / (TP + FN)

Question:

Out of all actual positives,
how many did we correctly identify?

---

Example

TP = 80

FN = 20

Recall:

80 / (80 + 20)

= 0.80

= 80%

---

# F1 Score

Formula:

2 × Precision × Recall
----------------------
 Precision + Recall

---

Why?

Balances Precision and Recall.

---

# When to Use?

Precision:

Spam Detection

Avoid False Positives

---

Recall:

Disease Detection

Avoid False Negatives

---

F1 Score:

Need balance between both

---

# Conclusion

Precision, Recall, and F1 Score provide a much better evaluation than accuracy alone.