#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


pip install pandas scikit-learn imbalanced-learn


# In[7]:


# TASK 5:-  CREDIT CARD FRAUD DETECTION

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

# Step 2: Load the dataset
file_path = r"C:\Users\jeeva\Downloads\creditcard.csv"  # Adjust this path to your dataset
data = pd.read_csv(file_path)

# Step 3: Explore the data
print("Data Overview:")
print(data.head())
print(data.info())
print("Class distribution before SMOTE:")
print(data['Class'].value_counts())

# Step 4: Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# Step 5: Prepare the data
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target variable

# Step 6: Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Step 8: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 9: Train the Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Step 10: Train the Random Forest model with adjustments
rf_model = RandomForestClassifier(n_estimators=10, max_depth=10, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)

# Step 11: Evaluate the Logistic Regression model
y_pred_logistic = logistic_model.predict(X_test)
print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logistic))
print(classification_report(y_test, y_pred_logistic))

# Step 12: Evaluate the Random Forest model
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Step 13: Visualizing Confusion Matrices
def plot_confusion_matrix(cm, title='Confusion Matrix'):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

# Confusion matrix for Logistic Regression
cm_logistic = confusion_matrix(y_test, y_pred_logistic)
plot_confusion_matrix(cm_logistic, title='Logistic Regression Confusion Matrix')

# Confusion matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
plot_confusion_matrix(cm_rf, title='Random Forest Confusion Matrix')

# Step 14: Plotting ROC Curves
def plot_roc_curve(y_test, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Get the predicted probabilities
y_scores_logistic = logistic_model.predict_proba(X_test)[:, 1]
y_scores_rf = rf_model.predict_proba(X_test)[:, 1]

# Plot ROC curves
plt.figure(figsize=(10, 6))
plot_roc_curve(y_test, y_scores_logistic, 'Logistic Regression')
plot_roc_curve(y_test, y_scores_rf, 'Random Forest')
plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




