#!/usr/bin/env python
# coding: utf-8

# In[1]:


#TASK 4 :- SALES PREDICTION USING PYTHON
# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Step 2: Load the Dataset
# Load the Dataset
file_path = r"C:\Users\jeeva\Downloads\advertising.csv"  # Use any of the options above
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Step 3: Data Exploration
# Check the data structure and summary statistics
print(data.info())
print(data.describe())

# Visualize relationships
sns.pairplot(data)
plt.show()

# Step 4: Data Preprocessing
# Check for missing values
print(data.isnull().sum())

# Fill or drop missing values
data.fillna(data.mean(), inplace=True)  # Example: fill with mean

# Convert categorical variables using one-hot encoding if necessary
data = pd.get_dummies(data, drop_first=True)

# Step 5: Feature Selection
# Define features (X) and target (y)
X = data.drop('Sales', axis=1)  # Replace 'Sales' with your target column name
y = data['Sales']

# Step 6: Train-Test Split
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Model Training
# Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 8: Predictions
# Predictions
y_pred_linear = linear_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Step 9: Model Evaluation
# Evaluate Linear Regression
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print(f"Linear Regression - RMSE: {rmse_linear:.2f}, R²: {r2_linear:.2f}")

# Evaluate Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest - RMSE: {rmse_rf:.2f}, R²: {r2_rf:.2f}")

# Step 10: Visualization of Results
plt.figure(figsize=(12, 6))

# Linear Regression results
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_linear)
plt.plot(y_test, y_test, color='red', lw=2)  # Line for perfect predictions
plt.title('Linear Regression Predictions')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')

# Random Forest results
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf)
plt.plot(y_test, y_test, color='red', lw=2)
plt.title('Random Forest Predictions')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')

plt.tight_layout()
plt.show()


# In[ ]:




