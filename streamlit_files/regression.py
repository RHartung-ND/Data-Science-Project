#!/usr/bin/env python
# coding: utf-8

# ##### Logistic Regression
# 
# The goal of this file is to implement a Logistic Regression model. 

# In[17]:


# Columns for reference
columns = ['un_profitability', 'vote_average', 'vote_count', 'revenue', 'runtime', 'budget', 'popularity', 'actor_avg', 'actor_med', 'actor_dev', 'production_avg', 'production_med', 'production_dev', 'release_year', 'release_month', 'original_title_matches', 'profit', 'original_language_english', 'american_film', 'english_language']


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Load the data
df = pd.read_csv('movie-data/cleaned_analysis_data.csv')

# Drop columns that are directly related to profitability
df = df.drop(columns=['budget', 'profit', 'revenue'])

# Drop columns that are not encoded
df = df.drop(columns=["title", "release_date"])

# Define target and features
y = df['un_profitability']
X = df.drop(columns=['un_profitability'])

# Create a train/test split (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define and instantiate the Logistic Regression model.
lr_model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
lr_model.fit(X_train, y_train)

# Predict on the test set
y_pred = lr_model.predict(X_test)

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Additional metric: number of correct predictions
correct_predictions = (y_test == y_pred).sum()
total_predictions = len(y_test)
print(f"\nCorrectly predicted {correct_predictions} out of {total_predictions} instances")


# In[ ]:


from sklearn.model_selection import GridSearchCV

# 'C' is the inverse regularization strength, and we test both L1 and L2 penalties.
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],  
}

# Instantiate the logistic regression model (liblinear to support L1 penalty)
lr = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)

# Set up GridSearchCV with 5-fold cross-validation, optimizing for accuracy.
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Output the best parameters and best cross-validation score
print("Best Parameters from Grid Search:")
print(grid_search.best_params_)
print("\nBest CV Score:")
print(grid_search.best_score_)

# Evaluate the best estimator on the test set
best_lr = grid_search.best_estimator_
y_pred_grid = best_lr.predict(X_test)
print("\nClassification Report for the Grid Search Optimized Model:")
print(classification_report(y_test, y_pred_grid, zero_division=0))

