import streamlit as st
import pandas as pd

st.title("Looking at the IMDB Movie Dataset")

df = pd.read_csv("movie-data/Imdb_Movie_Dataset.csv")

num_rows, num_columns = df.shape
st.write(f"Number of Columns: {num_columns}")
st.write(f"Number of Rows: {num_rows}")

unique_column_values = df.nunique(0)
st.write(f"How many values in each column are unique: {unique_column_values}")


# Showing simple statistics the numerical column on the dataset
st.write("**Summary Statistics for Numerical Attributes**")
st.dataframe(df.describe())


