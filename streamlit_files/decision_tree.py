#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.tree import export_graphviz
import graphviz
import pydotplus
from IPython.display import Image


# In[3]:


df = pd.read_csv('movie-data/cleaned_analysis_data.csv')


# In[4]:


features = ['vote_average', 'vote_count', 'runtime', 'popularity',
            'actor_avg', 'actor_med', 'actor_dev', 'production_avg', 
            'production_med', 'production_dev', 'original_title_matches',
            'original_language_english', 'american_film', 'english_language']


# In[5]:


# Drop columns that are directly related to profitability
df = df.drop(columns=['budget', 'profit', 'revenue', 'id', 'release_year', 'release_month'])

# Drop columns that are not encoded
df = df.drop(columns=["title", "release_date"])

# Define target and features
y = df['un_profitability']
X = df.drop(columns=['un_profitability'])

# Split the data into training and testing sets (70:30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[6]:


dot_data = export_graphviz(clf, out_file=None,
                           feature_names=features,
                           class_names=[str(c) for c in sorted(y.unique())],
                           filled=True, rounded=True,
                           special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("figures/movie_decision_tree")
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

