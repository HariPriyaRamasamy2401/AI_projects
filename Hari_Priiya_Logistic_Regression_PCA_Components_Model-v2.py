#!/usr/bin/env python
# coding: utf-8

# Business Problem Description:
# For this example, we will be using the breast cancer dataset from the UCI Machine Learning Repository. The objective is to build a model that can accurately predict whether a tumor is malignant or benign based on various features such as the radius, texture, and smoothness of the tumor. This is an important problem as early detection of cancer can significantly improve patient outcomes and reduce healthcare costs.

# In[4]:


import pandas as pd

#Loading dataset 
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
cancer_dataset = pd.read_csv(url, header=None)

print(cancer_dataset.head())


# Feature Engineering and Exploratory Data Analysis (EDA):
# Let's start by importing the dataset and taking a look at the first few rows.

# In[5]:


# Remove the first column as it is an ID and not a feature
cancer_dataset = cancer_dataset.drop(columns=[0])


# In[6]:


# Map the M (malignant) and B (benign) labels to integers
cancer_dataset[1] = cancer_dataset[1].map({'M': 1, 'B': 0})


# Feature Importance: We can use the coefficients of the logistic regression model to identify which features have the strongest influence on the prediction. We can plot a bar chart to visualize the coefficients and their magnitudes. This can help us understand which features are the most important in predicting whether a tumor is malignant or benign.

# In[7]:


import matplotlib.pyplot as plt

# Plot the feature coefficients
coefs = lr.coef_[0]
features = data.columns[1:]
plt.bar(features, coefs)
plt.xticks(rotation=90)
plt.title("Feature Importance")
plt.show()


# In[8]:


# Spliting the dataset into features and labels
X = cancer_dataset.iloc[:, 1:]
y = cancer_dataset.iloc[:, 0]


# In[9]:


# Performing standard scaling on the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[10]:


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train the logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr.predict(X_test)


# In[12]:


# Calculate the accuracy of the model
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")


# The Use of PCA for Dimensionality Reduction:
# To reduce the dimensionality of the dataset, we can use PCA. This can help to improve the performance of the model and reduce overfitting.

# In[13]:


from sklearn.decomposition import PCA

# Perform PCA and retain 95% of the variance
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# In[14]:


# Train a new logistic regression model on the PCA-transformed data
lr_pca = LogisticRegression()
lr_pca.fit(X_train_pca, y_train)


# In[15]:


# Make predictions on the test set
y_pred_pca = lr_pca.predict(X_test_pca)


# PCA Components: We can also plot a scatter plot to visualize the first two principal components of the PCA-transformed data. This can help us understand how the data is distributed and whether the classes are separable in the transformed space.

# In[16]:


# Plot the first two PCA components
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='coolwarm')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Components")
plt.show()


# In[17]:


# Calculate the accuracy of the model
acc_pca = accuracy_score(y_test, y_pred_pca)
print(f"Accuracy with PCA: {acc_pca:.4f}")


# Model Comparison: Finally, we can compare the performance of the logistic regression model with and without PCA using a bar chart. This can help us understand whether PCA has improved the performance of the model.

# In[18]:


# Plot the model accuracies
accuracies = [acc, acc_pca]
labels = ['Logistic Regression', 'Logistic Regression with PCA']
plt.bar(labels, accuracies)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.ylim((0.9, 1.0))
plt.show()


# In[ ]:




