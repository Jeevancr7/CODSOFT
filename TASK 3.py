#!/usr/bin/env python
# coding: utf-8

# In[37]:


#task 3 :- IRIS FLOWER CLASSIFICATION
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[18]:


# Load the dataset
data = pd.read_csv("C:\\Users\\jeeva\\Downloads\\IRIS.csv")

# Display the first few rows of the dataset
print("🌸 Dataset Overview:")
print(data.head())
print("\n📊 Dataset Info:")
print(data.info())
print("\n📈 Dataset Description:")
print(data.describe())
print("\n📉 Class Distribution:")
print(data['species'].value_counts())


# In[19]:


# Prepare features and labels
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']


# In[20]:


# Step 1: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n📦 Training set size:", X_train.shape[0])
print("📦 Testing set size:", X_test.shape[0])


# In[21]:


# Step 2: Standardization of features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[22]:


# Step 3: Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)


# In[23]:


# Step 4: Make predictions
y_pred = knn.predict(X_test_scaled)


# In[24]:


# Step 5: Evaluate the model
print("\n🔍 Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print("\n📋 Classification Report:")
class_report = classification_report(y_test, y_pred)
print(class_report)

accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")

# Additional metrics
from sklearn.metrics import f1_score, precision_score, recall_score

f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"⭐ F1 Score: {f1:.4f}")
print(f"🔑 Precision: {precision:.4f}")
print(f"📈 Recall: {recall:.4f}")


# In[38]:


# Step 6: Visualization of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("🔮 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[39]:


# Step 7: Visualize Feature Distributions
plt.figure(figsize=(12, 6))
sns.pairplot(data, hue='species', markers=["o", "s", "D"])
plt.title("Feature Distributions by Species")
plt.show()


# In[31]:


# Step 8: Visualize Feature Importance
# Create a list to hold DataFrames for each species
feature_importance_list = []

# Calculate the mean values for each feature grouped by species
for species in np.unique(y):
    means = X_train[y_train == species].mean()
    # Create a DataFrame for the current species
    species_df = pd.DataFrame({
        'Feature': means.index,
        'Importance': means.values,
        'Species': species
    })
    feature_importance_list.append(species_df)

# Concatenate all the DataFrames into a single DataFrame
feature_importance = pd.concat(feature_importance_list, ignore_index=True)

# Plotting feature importance for each species
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', hue='Species', data=feature_importance)
plt.title(" Feature Importance by Species")
plt.show()


# In[ ]:





# In[ ]:




