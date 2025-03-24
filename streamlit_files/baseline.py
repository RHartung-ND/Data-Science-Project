# ##### Baseline
# 
# The goal of this file is to calculate the baseline prediction for the un profitability metric using, the always predict the majority label strategy.



# Columns for reference
columns = ['un_profitability', 'vote_average', 'vote_count', 'revenue', 'runtime', 'budget', 'popularity', 'actor_avg', 'actor_med', 'actor_dev', 'production_avg', 'production_med', 'production_dev', 'release_year', 'release_month', 'original_title_matches', 'profit', 'original_language_english', 'american_film', 'english_language']

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('movie-data/cleaned_analysis_data.csv')

# Drop columns that are directly related to profitability
df = df.drop(columns=['budget', 'profit', 'revenue'])

# Drop columns that are not helpful
df = df.drop(columns=["title", "release_date"])

# Define target and features
y = df['un_profitability']

X = df.drop(columns=['un_profitability'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

majority_label = y_train.mode()[0]
print("Majority is", majority_label)
y_pred = np.full(shape=y_test.shape, fill_value=majority_label)

# Evaluate the baseline predictor
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Display additional metrics: number of correct predictions out of total predictions
correct_predictions = (y_test == y_pred).sum()
total_predictions = len(y_test)
print(f"\nCorrectly predicted {correct_predictions} out of {total_predictions} instances")
print(f"\nAccuracy: {correct_predictions/total_predictions}")