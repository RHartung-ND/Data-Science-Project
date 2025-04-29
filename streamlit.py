import streamlit as st
import pandas as pd
import sys
sys.path.append('./streamlit_files')
from forest import forest_model



st.title("Looking at the IMDB Movie Dataset")

df = pd.read_csv("movie-data/cleaned_analysis_data.csv")

example_df = df.copy()
example_df.drop(columns=["id", "title", "release_date", "profit", "revenue", "budget"], inplace=True)
st.write("Data Preview", example_df.head())

st.subheader("Create a Fake Movie")

vote_average = example_df['vote_average'].mean()
vote_count = example_df['vote_count'].mean()
popularity = example_df['popularity'].mean()
un_profitability = 0
def convert_scale(x, original_min=-200, original_max=2000, target_min=1, target_max=10):
    return target_min + (x - original_min) * (target_max - target_min) / (original_max - original_min)

runtime = st.slider("Please enter the runtime for your movie (minutes):", 1, 300)
actor_quality = st.slider("Please enter the quality of actors for your movie:", min_value = 1.00, max_value = 10.00)
production_quality = st.slider("Please enter the quality of production for your movie:", min_value = 1.00, max_value = 10.00)

actor_avg      = convert_scale(actor_quality, 1, 10, 300, 1200)
actor_med      = convert_scale(actor_quality, 1, 10, -200, 2000)
actor_dev      = convert_scale(actor_quality, 0, 10, 0, 2000)
production_avg = convert_scale(production_quality, 1, 10, 300, 1200)
production_med = convert_scale(production_quality, 1, 10, -200, 2000)
production_dev = convert_scale(production_quality, 0, 10, 0, 2000)


release_year = st.slider("Please enter the release year of your movie:", min_value = 1900, max_value= 2024)
release_month = st.slider("Please enter the release month of your movie:", min_value = 1, max_value= 12)
original_title_matches = st.number_input("Please enter whether the original title matches for your movie:", min_value = 0, max_value= 1)
original_language_english = st.number_input("Please enter whether the movie's original language was English:", min_value = 0, max_value= 1)
american_film = st.number_input("Please enter whether you movie was made in America:", min_value = 0, max_value= 1)
english_language = st.number_input("Please enter whether the movie contains English:", min_value = 0, max_value= 1)



data = {"vote_average": vote_average, "vote_count": vote_count, "runtime": runtime, "popularity": popularity, "actor_avg": actor_avg, "actor_med": actor_med, "actor_dev": actor_dev, "production_avg": production_avg, "production_med": production_med, "production_dev": production_dev, "release_year": release_year, "release_month": release_month, "original_title_matches": original_title_matches, "un_profitability": un_profitability, "original_language_english": original_language_english, "american_film": american_film, "english_language": english_language}

user_df = pd.DataFrame(data, index=[0])


st.markdown("-----")
st.header("Test your data on our random forest model")

n_estimators = st.slider("Choose the number of estimators that you want", 1, 100)
max_depth = st.slider("Choose the maximum depth that you want", 1, 100)
random_state = st.slider("Choose the random state that you want", 1, 100)
model, feature_columns, accuracy, precision, recall, f1= forest_model(df, n_estimators, max_depth, random_state)

fake_df_encoded = pd.get_dummies(user_df)

# Ensure alignment with training features
for col in feature_columns:
    if col not in fake_df_encoded:
        fake_df_encoded[col] = 0
fake_df_encoded = fake_df_encoded[feature_columns]

prediction = model.predict(fake_df_encoded)[0]
proba = model.predict_proba(fake_df_encoded)[0]

if prediction:
    st.error("Prediction: Unprofitable")
else:
    st.success("Prediction: Profitable")
    
st.info(f"Confidence: {max(proba):.2f}")


st.markdown(rf"The random forest model is using {n_estimators} estimators, a max depth of {max_depth}, and a random state of {random_state}.\
            The Accuracy is :green[**$\LARGE {accuracy*100:.1f}\%$**]\
            The Precision is :green[**$\LARGE {precision*100:.1f}\%$**]\
            The Recall is :green[**$\LARGE {recall*100:.1f}\%$**]\
            The F1-Score is :green[**$\LARGE {f1*100:.1f}\%$**]""")