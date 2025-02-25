# import numpy as np
# import pandas as pd
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.preprocessing import StandardScaler

# def impute_zero(df, k=5, columns_to_impute=["revenue", "budget", "runtime"], feature_columns=["vote_average", "vote_count", "runtime", "popularity"]):
#     """Imputes missing values (0) with scaling."""
#     df_imputed = df.copy()
#     scaler = StandardScaler()
#     df_imputed[feature_columns] = scaler.fit_transform(df_imputed[feature_columns]) #scale feature columns

#     df_imputed["both_zero"] = (df_imputed["revenue"] == 0) & (df_imputed["budget"] == 0)

#     for target_col in columns_to_impute:
#         missing_indices = (df_imputed[target_col] == 0)
#         if missing_indices.any():
#             known_indices = ~missing_indices
#             known_data = df_imputed.loc[known_indices, feature_columns + [target_col]].dropna()
#             missing_data = df_imputed.loc[missing_indices, feature_columns]

#             if known_data.shape[0] < k:
#                 k_used = known_data.shape[0]
#             else:
#                 k_used = k

#             if known_data.shape[0] > 0:
#                 knn = KNeighborsRegressor(n_neighbors=k_used)
#                 knn.fit(known_data[feature_columns], known_data[target_col])
#                 imputed_values = knn.predict(missing_data[feature_columns])
#                 df_imputed.loc[missing_indices, target_col] = imputed_values
#             else:
#                 df_imputed.loc[missing_indices, target_col] = df_imputed[target_col].replace(0, np.nan).mean()
#     df_imputed[feature_columns] = scaler.inverse_transform(df_imputed[feature_columns]) #inverse transform feature columns.
#     return df_imputed

# # Example usage (replace with your actual movie data):
# if __name__ == "__main__":
#     data = {
#         "vote_average": [6.1, 6.1, 7.332, 5.163, 6.721],
#         "vote_count": [252, 172, 128, 127, 77],
#         "revenue": [1, 3, 3, 2, 3],
#         "runtime": [117, 92, 84, 90, 110],
#         "budget": [5, 4860000, 162297, 3627000, 10230236],
#         "popularity": [20.436, 12.674, 4.615, 4.963, 7.203],
#     }
#     movie_df = pd.DataFrame(data)

#     imputed_movie_df = impute_zero(movie_df)
#     print("Original DataFrame:\n", movie_df)
#     print("\nImputed DataFrame:\n", imputed_movie_df)


import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

def impute_zero(df, k=5, feature_columns=["vote_average", "vote_count", "popularity"]):
    """
    Imputes revenue, budget (conditionally), and runtime.

    Args:
        df (pandas.DataFrame): The movie DataFrame.
        k (int): The number of neighbors.
        feature_columns (list): Columns used as features.
    """
    df_imputed = df.copy()
    scaler = StandardScaler()
    df_imputed[feature_columns] = scaler.fit_transform(df_imputed[feature_columns])

    # 1. Impute Revenue
    if "revenue" in df_imputed.columns:
        missing_revenue = df_imputed["revenue"] == 0
        if missing_revenue.any():
            known_revenue = df_imputed.loc[~missing_revenue, feature_columns + ["revenue"]].dropna()
            missing_revenue_data = df_imputed.loc[missing_revenue, feature_columns]

            if known_revenue.shape[0] >= k:
                knn_revenue = KNeighborsRegressor(n_neighbors=k)
                knn_revenue.fit(known_revenue[feature_columns], known_revenue["revenue"])
                df_imputed.loc[missing_revenue, "revenue"] = knn_revenue.predict(missing_revenue_data[feature_columns])
            else:
                df_imputed.loc[missing_revenue, "revenue"] = df_imputed["revenue"].replace(0, np.nan).mean()

    # 2. Impute Budget (Conditionally)
    if "budget" in df_imputed.columns:
        missing_budget = df_imputed["budget"] == 0
        if missing_budget.any():
            known_budget = df_imputed.loc[~missing_budget, feature_columns + ["budget"]].dropna()
            missing_budget_data = df_imputed.loc[missing_budget, feature_columns]

            if known_budget.shape[0] >= k:
                knn_budget = KNeighborsRegressor(n_neighbors=k)
                knn_budget.fit(known_budget[feature_columns], known_budget["budget"])
                df_imputed.loc[missing_budget, "budget"] = knn_budget.predict(missing_budget_data[feature_columns])
            else:
                df_imputed.loc[missing_budget, "budget"] = df_imputed["budget"].replace(0, np.nan).mean()

    # 3. Impute Runtime
    if "runtime" in df_imputed.columns:
        missing_runtime = df_imputed["runtime"] == 0
        if missing_runtime.any():
            known_runtime = df_imputed.loc[~missing_runtime, feature_columns + ["runtime"]].dropna()
            missing_runtime_data = df_imputed.loc[missing_runtime, feature_columns]

            if known_runtime.shape[0] >= k:
                knn_runtime = KNeighborsRegressor(n_neighbors=k)
                knn_runtime.fit(known_runtime[feature_columns], known_runtime["runtime"])
                df_imputed.loc[missing_runtime, "runtime"] = knn_runtime.predict(missing_runtime_data[feature_columns])
            else:
                df_imputed.loc[missing_runtime, "runtime"] = df_imputed["runtime"].replace(0, np.nan).mean()

    df_imputed[feature_columns] = scaler.inverse_transform(df_imputed[feature_columns])
    return df_imputed

# Example usage:
if __name__ == "__main__":
    data = {
        "vote_average": [6.1, 6.1, 7.332, 5.163, 6.721],
        "vote_count": [252, 172, 128, 127, 77],
        "revenue": [1, 0, 0, 2, 0],
        "runtime": [117, 0, 84, 90, 0],
        "budget": [5, 0, 239482, 3627000, 0],
        "popularity": [20.436, 12.674, 4.615, 4.963, 7.203],
    }
    movie_df = pd.DataFrame(data)

    imputed_movie_df = impute_zero(movie_df)
    print("Original DataFrame:\n", movie_df)
    print("\nImputed DataFrame:\n", imputed_movie_df)