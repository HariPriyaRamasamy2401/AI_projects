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

