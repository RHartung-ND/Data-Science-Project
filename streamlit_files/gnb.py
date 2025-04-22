#!/usr/bin/env python
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def gnb_model(df, state):
    df.drop(columns=["id", "title", "release_date", "profit", "revenue", "budget"], inplace=True)
    Y_data = df["un_profitability"]
    X_data = df.drop(columns=["un_profitability"])
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.20, random_state=state)
    naive_bayes_model = GaussianNB()
    naive_bayes_model.fit(X_train, Y_train)
    Y_prediction = naive_bayes_model.predict(X_test)
    return accuracy_score(Y_test, Y_prediction)


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("movie-data/cleaned_analysis_data.csv")
    gnb_model(df)