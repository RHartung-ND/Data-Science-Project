#!/usr/bin/env python
# coding: utf-8

# ##### K-Nearest Neighbor (KNN)
# 
# The goal of this file is to implement a KNN model. 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def knn_model(df, n_neighbors):
    df = df.drop(columns=['budget', 'profit', 'revenue', 'title', 'release_date'])
    y = df['un_profitability']
    X = df.drop(columns=['un_profitability'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)  # You can adjust n_neighbors
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy