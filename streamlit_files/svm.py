#!/usr/bin/env python
# coding: utf-8

# ##### Support Vector Machine
# 
# The goal of this file is to implement a Support Vector Machine model. 

# In[17]:


# Columns for reference
columns = ['un_profitability', 'vote_average', 'vote_count', 'revenue', 'runtime', 'budget', 'popularity', 'actor_avg', 'actor_med', 'actor_dev', 'production_avg', 'production_med', 'production_dev', 'release_year', 'release_month', 'original_title_matches', 'profit', 'original_language_english', 'american_film', 'english_language']


# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('movie-data/cleaned_analysis_data.csv')
df = df.drop(columns=['budget', 'profit', 'revenue', 'title', 'release_date'])
y = df['un_profitability']
X = df.drop(columns=['un_profitability'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

svm_model = SVC(C=1,class_weight='balanced', random_state=42)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

