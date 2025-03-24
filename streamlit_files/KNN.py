#!/usr/bin/env python
# coding: utf-8

# ##### K-Nearest Neighbor (KNN)
# 
# The goal of this file is to implement a KNN model. 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

features = ['vote_average', 'vote_count', 'runtime', 'popularity',
            'actor_avg', 'actor_med', 'actor_dev', 'production_avg', 
            'production_med', 'production_dev', 'original_title_matches',
            'original_language_english', 'american_film', 'english_language']

# Drop columns that are directly related to profitability

def knn_model(df, n_neighbors):

    df = df.drop(columns=['budget', 'profit', 'revenue', 'title', 'release_date'])
    # Define target and features
    y = df['un_profitability']
    X = df.drop(columns=['un_profitability'])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets (70:30 split)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Create and train the KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)  # You can adjust n_neighbors
    knn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
