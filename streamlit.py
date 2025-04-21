import streamlit as st
import pandas as pd
import sys
import numpy as np
sys.path.append('./streamlit_files')


from KNN import knn_model
from decision_tree import decision_tree_model
from forest import forest_model
from gnb import gnb_model

st.title("Looking at the IMDB Movie Dataset")

# df = pd.read_csv("movie-data/cleaned_analysis_data.csv")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    example_df = df.copy()
    example_df.drop(columns=["id", "title", "release_date", "profit", "revenue", "budget"], inplace=True)
    st.write("Data Preview", example_df.head())


    st.write("Data Preview", example_df.columns)

    st.write(example_df['release_month'].min())


    st.subheader("Create a Fake Movie")

    vote_average = st.slider("Please enter the average voting score for your movie:",0.00,10.00)
    vote_count = st.slider("Please enter the number of votes for your movie:", min_value = 1, max_value = 40000)
    runtime = st.number_input("Please enter the runtime for your movie:", 1, 1000)
    popularity = st.number_input("Please enter the popularity of your movie:")
    actor_avg = st.slider("Please enter the average actor score for your movie:", min_value = -200, max_value = 2000)
    actor_med = st.slider("Please enter the median actor score for your movie:", min_value = -200, max_value = 2000)
    actor_dev = st.slider("Please enter the standard deviation actor score for your movie:", min_value = 0, max_value = 2000)
    production_avg = st.slider("Please enter the average production score for your movie:", min_value = -200, max_value = 2000)
    production_med = st.slider("Please enter the median production score for your movie:", min_value = -200, max_value = 2000)
    production_dev = st.slider("Please enter the standard deviation production score for your movie:", min_value = 0, max_value = 2000)
    release_year = st.number_input("Please enter the release year of your movie:", min_value = 1900, max_value= 2024)
    release_month = st.slider("Please enter the release month of your movie:", min_value = 1, max_value= 12)
    original_title_matches = st.number_input("Please enter whether the original title matches for your movie:", min_value = 0, max_value= 1)
    un_profitability = st.number_input("Please enter the un_profitability for your movie:", min_value = 0, max_value= 1)
    original_language_english = st.number_input("Please enter whether the movie's original language was English:", min_value = 0, max_value= 1)
    american_film = st.number_input("Please enter whether you movie was made in America:", min_value = 0, max_value= 1)
    english_language = st.number_input("Please enter whether the movie contains English:", min_value = 0, max_value= 1)


    data = {"vote_average": vote_average, "vote_count": vote_count, "runtime": runtime, "popularity": popularity, "actor_avg": actor_avg, "actor_med": actor_med, "actor_dev": actor_dev, "production_avg": production_avg, "production_med": production_med, "production_dev": production_dev, "release_year": release_year, "release_month": release_month, "original_title_matches": original_title_matches, "un_profitability": un_profitability, "original_language_english": original_language_english, "american_film": american_film, "english_language": english_language}
    

    model_choice = st.selectbox("Choose a machine learning model", [
            "None",
            "K-Nearest Neighbor",
            "Decision Tree",
            "Random Forest",
            "Gaussian Naive Bayes"
        ])


    if model_choice == "K-Nearest Neighbor":
        st.header("K-Nearest Neighbor")
        n_neighbors = st.slider("Choose how many neighbors you want", 1, 100)
        knn_acc = knn_model(df, n_neighbors) * 100
        st.markdown(rf"The accuracy of the KNN model with {n_neighbors} neighbors is: :green[**$\LARGE {knn_acc:.1f}\%$**]")

    elif model_choice == "Decision Tree":
        st.header("Decision Tree")
        state = st.slider("Choose the random state that you want", 1, 100)
        dec_tree_acc = decision_tree_model(df, state) * 100
        st.markdown(rf"The accuracy of the Decision Tree model with a random state of {state} is: :green[**$\LARGE {dec_tree_acc:.1f}\%$**]")

    elif model_choice == "Random Forest":
        st.header("Random Forest Model")
        n_estimators = st.slider("Choose the number of estimators that you want", 1, 100)
        max_depth = st.slider("Choose the maximum depth that you want", 1, 100)
        random_state = st.slider("Choose the random state that you want", 1, 100)
        forest_acc = forest_model(df, n_estimators, max_depth, random_state)*100
        st.markdown(rf"The accuracy of the Decision Tree model with {n_estimators} estimators, a max depth of {max_depth}, and a random state of {random_state} is: :green[**$\LARGE {forest_acc:.1f}\%$**]")

    elif model_choice == "Gaussian Naive Bayes":
        st.header("Gaussian Naive Bayes")
        state = st.slider("Choose the random state that you want", 1, 100)
        gnb_tree_acc = gnb_model(df, state) * 100
        st.markdown(rf"The accuracy of the Gaussian Naive Bayes model with a random state of {state} is: :green[**$\LARGE {gnb_tree_acc:.1f}\%$**]")




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