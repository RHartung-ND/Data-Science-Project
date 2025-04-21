#!/usr/bin/env python
# coding: utf-8
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def decision_tree_model(df, state):
    df = df.drop(columns=['budget', 'profit', 'revenue', 'id', 'release_year', 'release_month'])
    df = df.drop(columns=["title", "release_date"])
    y = df['un_profitability']
    X = df.drop(columns=['un_profitability'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=state)
    clf = DecisionTreeClassifier(random_state=state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
