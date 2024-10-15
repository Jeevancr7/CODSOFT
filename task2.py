#!/usr/bin/env python
# coding: utf-8

# In[27]:


#TASK 2 :- MOVIE RATING PREDICTION WITH PYTHON


# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# Step 2: Load the dataset
file_path = r"C:\Users\jeeva\Downloads\advertising.csv"
data = pd.read_csv(file_path)

# Step 3: Data Preprocessing
print("First few rows of the dataset:")
print(data.head())
print("\nDataset Info:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())

# Check for missing values and drop them
data = data.dropna()

# Step 4: Exploratory Data Analysis (EDA)
plt.figure(figsize=(12, 6))
sns.pairplot(data, diag_kind='kde')
plt.suptitle('Pairplot of Advertising Data', y=1.02)
plt.show()

# Step 5: Correlation Matrix
plt.figure(figsize=(8, 6))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Step 6: Split into Features (X) and Target (y)
X = data[['TV', 'Radio', 'Newspaper']]  # Using existing feature columns
y = data['Sales']  # Target variable

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Make Predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate the Model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'\nModel Evaluation:')
print(f'RMSE: {rmse:.2f}')
print(f'R^2: {r2:.2f}')

# Visualize Actual vs Predicted Sales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  # Perfect prediction line
plt.title('Actual vs Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.grid(True)
plt.show()

# Step 11: Feature Importance
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 6))
plt.title('Feature Importance')
plt.barh(range(X.shape[1]), importances[indices], align='center')
plt.yticks(range(X.shape[1]), feature_names[indices])
plt.xlabel('Relative Importance')
plt.show()

# Step 12: Distribution of Errors
errors = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(errors, kde=True, bins=20)
plt.title('Distribution of Prediction Errors')
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Step 13: Summary of Results
summary_df = pd.DataFrame({
    'Actual Sales': y_test,
    'Predicted Sales': y_pred,
    'Error': errors
})
print("\nSummary of Actual vs Predicted Sales:")
print(summary_df.head(10))


# In[ ]:





# In[ ]:





# In[ ]:




