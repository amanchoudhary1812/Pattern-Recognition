import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load the dataset from the CSV file (replace 'path_to_your_file.csv' with your actual file path)
data = pd.read_csv('path_to_your_file.csv')

# Print out the column names to inspect
print("Column names in dataset:")
print(data.columns)

# Clean column names by removing any extra spaces (if necessary)
data.columns = data.columns.str.strip()

# Print first few rows to understand the structure of the data
print("First few rows of the dataset:")
print(data.head())

# Replace 'x_value' and 'y_value' with actual column names from the dataset
# Example: Let's assume the independent variable column is 'X_column' and dependent 'Y_column'
X = data['X_column'].values.reshape(-1, 1)  # Independent variable
y = data['Y_column'].values  # Dependent variable

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

# Non-linear curve fitting (example using quadratic function)
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

params, _ = curve_fit(quadratic, X.flatten(), y)
a, b, c = params
y_pred_nonlinear = quadratic(X.flatten(), a, b, c)

# Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE)
mse_linear = mean_squared_error(y, y_pred_linear)
mae_linear = mean_absolute_error(y, y_pred_linear)

mse_nonlinear = mean_squared_error(y, y_pred_nonlinear)
mae_nonlinear = mean_absolute_error(y, y_pred_nonlinear)

# Display results
print("Linear Regression Results:")
print(f"  Coefficients: {linear_model.coef_}")
print(f"  Intercept: {linear_model.intercept_}")
print(f"  MSE: {mse_linear:.3f}")
print(f"  MAE: {mae_linear:.3f}")

print("\nNon-linear Regression Results:")
print(f"  Parameters (a, b, c): {params}")
print(f"  MSE: {mse_nonlinear:.3f}")
print(f"  MAE: {mae_nonlinear:.3f}")

# Plotting the data and the regression lines
plt.figure(figsize=(12, 6))

# Plot original data
plt.scatter(X, y, color="black", label="Original Data")

# Linear regression line
plt.plot(X, y_pred_linear, color="blue", label="Linear Regression")

# Non-linear regression curve
plt.plot(X, y_pred_nonlinear, color="red", label="Non-linear Regression (Quadratic)")

# Adding labels and title
plt.title("Regression Analysis")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
