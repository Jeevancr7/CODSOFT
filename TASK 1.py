#!/usr/bin/env python
# coding: utf-8

# In[1]:


# TASK 1: TITANIC SURVIVAL PREDICTION

# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the Titanic dataset
dataset_path = "C:/Users/jeeva/Downloads/Titanic-Dataset.csv"  # Path to the dataset
data = pd.read_csv(dataset_path)

# Step 2: Explore the dataset
print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# Step 3: Visualize Survival Counts
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=data)
plt.title('Survival Count on the Titanic')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Step 4: Data Cleaning and Preparation
# Filling missing values
data['Age'].fillna(data['Age'].mean(), inplace=True)  # Fill missing 'Age' with mean
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)  # Fill 'Embarked' with mode

# Dropping unnecessary columns
data.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

# Encoding categorical variables
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])  # Encode 'Sex' (0 for female, 1 for male)
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)  # One-hot encoding 'Embarked'

# Step 5: Feature Selection
X = data.drop('Survived', axis=1)  # Features
y = data['Survived']  # Target variable

# Step 6: Splitting Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Building the Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 8: Making Predictions
y_pred = rf_model.predict(X_test)

# Step 9: Model Evaluation
print("\nModel Evaluation:")
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 10: Feature Importance Analysis
feature_importances = pd.DataFrame(rf_model.feature_importances_, 
                                    index=X_train.columns, 
                                    columns=['Importance']).sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importances)

# Optional: Visualizing Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances['Importance'], y=feature_importances.index)
plt.title('Feature Importance in Titanic Survival Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()


# In[ ]:




