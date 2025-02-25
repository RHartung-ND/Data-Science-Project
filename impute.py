import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

def impute_zero(df, k=5, columns_to_impute=["revenue", "budget", "runtime"], feature_columns=["vote_average", "vote_count", "runtime", "popularity"]):
    """Imputes missing values (0) with scaling."""
    df_imputed = df.copy()
    scaler = StandardScaler()
    df_imputed[feature_columns] = scaler.fit_transform(df_imputed[feature_columns]) #scale feature columns

    for target_col in columns_to_impute:
        missing_indices = (df_imputed[target_col] == 0)
        if missing_indices.any():
            known_indices = ~missing_indices
            known_data = df_imputed.loc[known_indices, feature_columns + [target_col]].dropna()
            missing_data = df_imputed.loc[missing_indices, feature_columns]

            if known_data.shape[0] < k:
                k_used = known_data.shape[0]
            else:
                k_used = k

            if known_data.shape[0] > 0:
                knn = KNeighborsRegressor(n_neighbors=k_used)
                knn.fit(known_data[feature_columns], known_data[target_col])
                imputed_values = knn.predict(missing_data[feature_columns])
                df_imputed.loc[missing_indices, target_col] = imputed_values
            else:
                df_imputed.loc[missing_indices, target_col] = df_imputed[target_col].replace(0, np.nan).mean()
    df_imputed[feature_columns] = scaler.inverse_transform(df_imputed[feature_columns]) #inverse transform feature columns.
    return df_imputed

# Example usage (replace with your actual movie data):
if __name__ == "__main__":
    data = {
        "vote_average": [6.1, 6.1, 7.332, 5.163, 6.721],
        "vote_count": [252, 172, 128, 127, 77],
        "revenue": [1, 3, 3, 2, 3],
        "runtime": [117, 92, 84, 90, 110],
        "budget": [5, 4860000, 162297, 3627000, 10230236],
        "popularity": [20.436, 12.674, 4.615, 4.963, 7.203],
    }
    movie_df = pd.DataFrame(data)

    imputed_movie_df = impute_zero(movie_df)
    print("Original DataFrame:\n", movie_df)
    print("\nImputed DataFrame:\n", imputed_movie_df)