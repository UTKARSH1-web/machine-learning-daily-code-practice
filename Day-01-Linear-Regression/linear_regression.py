from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

#Input data
X = np.array([[1], [2], [3], [4], [5]])

#Output data
y = np.array([2, 4, 6, 8, 10])

#Create a linear regression model
model = LinearRegression()

#Fit the model to the data
model.fit(X, y)

#Predict the output for new input
predicted = model.predict([[6]])
print("Prediction:", predicted)

#Graph Visualization
plt.scatter(X, y, color='blue', label='Data Points')

#best fit line
plt.plot(X, model.predict(X), color='red', label='Best Fit Line')
plt.xlabel('Input (X)')
plt.ylabel('Output (y)')
plt.title('Linear Regression')
plt.show()
