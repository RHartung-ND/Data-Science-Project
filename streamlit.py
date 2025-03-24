import streamlit as st
import pandas as pd
import sys
sys.path.append('./streamlit_files')


from KNN import knn_model

st.title("Looking at the IMDB Movie Dataset")

df = pd.read_csv("movie-data/cleaned_analysis_data.csv")


# st.write('''
# - Baseline Model: Calculates baseline predictions of 64%
# - Decision Tree Model: Creates a decision tree model, accuracy of 75%
# - Forest Model: uses random forest model, accuracy of 83%
# - Gradient Boost Model: uses two gradient boost models, gets 84%
# - Regression Model: uses logistic regression, gets 65%
# - Gaussian Naive Bayes Model: Uses GaussianNB to prediction 45%
# - SVM Model: Uses support vector machine, gets 53%
# - Neural Network Model: Implemented a basic neural network, still need to work on it, but it gets 78%
# - KNN Model: uses K-nearest neighbor, gets 70%
#          ''')


st.header("K-Nearest Neighbor")
n_neighbors = st.slider("Choose how many neighbors you want", 1, 20)
knn_acc = knn_model(df, n_neighbors) * 100

st.markdown(f"The accuracy of the KNN model with {n_neighbors} neighbors is: :green[**$\LARGE {knn_acc:.1f}\%$**]")
