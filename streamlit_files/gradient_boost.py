#!/usr/bin/env python
# coding: utf-8

# ##### Gradient Boost
# 
# The goal of this file is to implement a gradient boost model. 

# In[1]:


# Columns for reference
columns = ['un_profitability', 'vote_average', 'vote_count', 'revenue', 'runtime', 'budget', 'popularity', 'actor_avg', 'actor_med', 'actor_dev', 'production_avg', 'production_med', 'production_dev', 'release_year', 'release_month', 'original_title_matches', 'profit', 'original_language_english', 'american_film', 'english_language']


# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('movie-data/cleaned_analysis_data.csv')

# Drop columns that are directly related to profitability
df = df.drop(columns=['budget', 'profit', 'revenue'])

# Drop columns that are not encoded
df = df.drop(columns=["title", "release_date"])


# Define target and features
y = df['un_profitability']
X = df.drop(columns=['un_profitability'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define key parameters for the Gradient Boosting model
n_estimators = 100      # Number of boosting stages to perform
max_depth = 6           # Maximum depth of the individual regression estimators
learning_rate = 0.1     # Learning rate shrinks the contribution of each tree
random_state = 42       # Seed for reproducibility

gb_model = GradientBoostingClassifier(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      learning_rate=learning_rate,
                                      random_state=random_state)

gb_model.fit(X_train, y_train)
y_pred = gb_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

correct_predictions = (y_test == y_pred).sum()
total_predictions = len(y_test)
print(f"\nCorrectly predicted {correct_predictions} out of {total_predictions} instances")


# In[3]:


from sklearn.model_selection import GridSearchCV

param_grid_gb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

gb = GradientBoostingClassifier(random_state=42)
grid_search_gb = GridSearchCV(gb, param_grid_gb, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_gb.fit(X_train, y_train)

print("Best GB Params:", grid_search_gb.best_params_)
print("Best GB CV Score:", grid_search_gb.best_score_)

best_gb = grid_search_gb.best_estimator_
y_pred_gb = best_gb.predict(X_test)
print(classification_report(y_test, y_pred_gb, zero_division=0))


# In[6]:


# Using XGBoost as an alternative gradient boosting method
from xgboost import XGBClassifier
xgb_model = XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.3, random_state=42,
                         eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
correct_predictions = (y_test == y_pred).sum()
total_predictions = len(y_test)
print(f"\nCorrectly predicted {correct_predictions} out of {total_predictions} instances")


# In[7]:


param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_xgb.fit(X_train, y_train)

print("Best XGB Params:", grid_search_xgb.best_params_)
print("Best XGB CV Score:", grid_search_xgb.best_score_)

best_xgb = grid_search_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
print(classification_report(y_test, y_pred_xgb, zero_division=0))

