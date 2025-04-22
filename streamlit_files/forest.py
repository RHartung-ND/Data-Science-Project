import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

def forest_model(df, n_estimators = 100, max_depth = 80, random_state = 42):
    df = df.drop(columns=['budget', 'profit', 'revenue'])
    df = df.drop(columns=["title", "release_date"])
    y = df['un_profitability']
    X = df.drop(columns=['un_profitability'])

    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    class_weight = 'balanced'  # Handle imbalance by weighting classes

    rf_model = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    random_state=random_state,
                                    class_weight=class_weight)

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    accuracy = (y_test == y_pred).sum() / len(y_test)

    return rf_model, accuracy, X.columns


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("movie-data/cleaned_analysis_data.csv")
    forest_model(df)