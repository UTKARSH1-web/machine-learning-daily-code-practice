# Linear Regression Theory

## Overview
Linear Regression is a fundamental supervised learning algorithm used for predicting continuous values. It models the relationship between input features and a target variable using a linear equation.

## Mathematical Formulation
The linear regression model can be expressed as:

$$y = mx + b$$

Or in multivariate form:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n$$

Where:
- `y` is the predicted value (dependent variable)
- `x_i` are the input features (independent variables)
- `\beta_i` are the coefficients/weights
- `\beta_0` is the bias term (intercept)

## Cost Function
The Mean Squared Error (MSE) is commonly used as the cost function:

$$MSE = \frac{1}{m} \sum_{i=1}^{m} (y_{pred}^{(i)} - y_{true}^{(i)})^2$$

Where `m` is the number of samples.

## Optimization
- **Gradient Descent**: Iteratively updates weights to minimize the cost function
- **Normal Equation**: Direct solution for optimal weights (for small datasets)

## Advantages
- Simple and interpretable
- Computationally efficient
- Works well for linearly separable data
- Good baseline model

## Disadvantages
- Assumes linear relationship between features and target
- Sensitive to outliers
- May underfit complex patterns
- Limited for high-dimensional data

## Use Cases
- Housing price prediction
- Sales forecasting
- Stock price prediction
- Trend analysis
