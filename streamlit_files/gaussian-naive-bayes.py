#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


# In[9]:


df = pd.read_csv('movie-data/cleaned_analysis_data.csv')

print(df.columns)

# Removing columns which aren't needed (or directly related to un-profitability).
df.drop(columns=["id", "title", "release_date", "profit", "revenue", "budget"], inplace=True)

# Sorting the data by release date. (lower indexes are earlier dates)
# df.sort_values(by="release_year", inplace=True)

print(df.columns)
# print(df.isna().sum())


# In[17]:


Y_data = df["un_profitability"]
X_data = df.drop(columns=["un_profitability"])

# Creating an 80/20 split 
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.20, random_state=14)
print(f"X_train size: {X_train.shape}")
print(f"X_test size: {X_test.shape}")

# majority_split = round(df.shape[0] * 0.80)
# minority_split = round(df.shape[0] * 0.20)

# X_data = df.drop(columns=["un_profitability"]).values
# Y_data = df["un_profitability"].values

# X_train = X_data[:majority_split]
# Y_train = Y_data[:majority_split]

# X_test = X_data[majority_split:]
# Y_test = Y_data[majority_split:]


# In[23]:


naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, Y_train)

Y_prediction = naive_bayes_model.predict(X_test)

print("Classification Report:\n", classification_report(Y_test, Y_prediction))

# accuracy = accuracy_score(Y_test, Y_prediction)
# precision = precision_score(Y_test, Y_prediction)
# recall = recall_score(Y_test, Y_prediction)
# f1 = f1_score(Y_test, Y_prediction)

# print(f"Accuracy: {accuracy:.2f}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1 Score: {f1:.2f}")


