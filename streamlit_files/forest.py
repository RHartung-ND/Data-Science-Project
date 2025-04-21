from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def forest_model(df, n_estimators = 100, max_depth = 80, random_state = 42):
    df = df.drop(columns=['budget', 'profit', 'revenue'])
    df = df.drop(columns=["title", "release_date"])
    y = df['un_profitability']
    X = df.drop(columns=['un_profitability'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    class_weight = 'balanced'  # Handle imbalance by weighting classes

    rf_model = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    random_state=random_state,
                                    class_weight=class_weight)

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix:")
    # print(cm)

    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred, zero_division=0))

    # Display additional metrics: number of correct predictions out of total predictions
    correct_predictions = (y_test == y_pred).sum()
    total_predictions = len(y_test)
    # print(f"\nCorrectly predicted {correct_predictions} out of {total_predictions} instances")
    return correct_predictions/total_predictions
    # print(f"Accuracy: {(correct_predictions/total_predictions)*100}")






    # # Define a parameter grid for the Random Forest model.
    # param_grid_rf = {
    #     'n_estimators': [100, 200, 300],      # Number of trees in the forest.
    #     'max_depth': [None, 10, 20, 80],        # Maximum depth of each tree.
    #     'min_samples_split': [2, 5, 10],        # Minimum number of samples required to split an internal node.
    #     'class_weight': ['balanced', None]      # Handle class imbalance.
    # }


    # rf = RandomForestClassifier(random_state=42)

    # # Set up GridSearchCV with 5-fold cross-validation.
    # grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
    # grid_search_rf.fit(X_train, y_train)

    # # Output the best parameters and best cross-validation score.
    # print("Best Parameters for Random Forest Grid Search:")
    # print(grid_search_rf.best_params_)
    # print("\nBest CV Score:")
    # print(grid_search_rf.best_score_)

    # # Evaluate the best estimator on the test set.
    # best_rf = grid_search_rf.best_estimator_
    # y_pred_rf = best_rf.predict(X_test)
    # print("\nClassification Report for the Grid Search Optimized Random Forest:")
    # print(classification_report(y_test, y_pred_rf, zero_division=0))


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("movie-data/cleaned_analysis_data.csv")
    forest_model(df)