# Data-Science-Project

## Getting the data

1. Download the [principals dataset](https://datasets.imdbws.com/title.principals.tsv.gz), the [name dataset](https://datasets.imdbws.com/name.basics.tsv.gz), and the [general movies dataset](https://www.kaggle.com/datasets/anandshaw2001/imdb-data).

3. Extract the files using either 7zip, Winrar, or gzip. For the IMDb dataset, rename it to `Imdb_Movie_Dataset.csv`

4. Move the files into a folder called `movie-data` in the root directory of the project

5. Install the Python packages using the following command: `pip install -r requirements.txt` (Note: There are many of them)

6. Run the following Python notebooks in order:
    
    ### Preprocessing
    1. **combine-data.ipynb** - Extracts the data, combines it with cast & crew, and loads it into a csv file (produces combined_df.csv).
    
    2. **pivot-data.ipynb** - Calculates a score for each actor and crew member based on the amount of money the movies they were in made. Compiles score averages for each movie and loads it into a csv file (produces movies_with_scores.csv). It also pivots the data in a much more basic way (just to make looking at data easier) in a csv file (produces basic_pivot_df.csv). 

    3. **encode-data.ipynb** - Drops imdb_id, overview, tagline, production_companies, keywords, and status. Converts release date into a date, month, and year. One-hot encodes adult, original language, genres, production countries, and spoken languages. Saves it all in one dataframe and onto a csv file (produces movies_encoded.csv).

    ### Analysis

    4. **analysis.ipynb** - Displays statistics on the data such as counts, averages, standard deviations, and correlations. Calculated profit and unprofitability scores and calculated correlations with non-profitability. Also calculates baseline predictions at 64%. Saved final dataset containing highest correlations as a csv (cleaned_analysis_data.csv).

## The different machine learning models

To run one of the various machine learning models, open the corresponding notebook and run every single cell. 

| File                        | Model                                                   | Accuracy |
|----------------------------|----------------------------------------------------------|----------|
| gaussian-naive-bayes.ipynb | GaussianNB                                               | 45%      |
| svm.ipynb                  | Support Vector Machine                                   | 53%      |
| baseline.ipynb            | Calculates baseline confusion matrix predictions on 50/50 splits and 80/20 splits. | 64%      |
| regression.ipynb          | Logistic regression                                      | 65%      |
| KNN.ipynb                  | K-nearest neighbor                                       | 70%      |
| decision_tree.ipynb       | Decision Tree model                                      | 75%      |
| neural.ipynb              | Basic neural network                                     | 78%      |
| forest.ipynb              | Random forest model                                      | 84%      |
| gradient_boost.ipynb      | Two gradient boost models                                | 84%      |


## Streamlit application

To run the Streamlit application, type the following command into the terminal: `streamlit run demo.py`. This should run the application on your local machine in your default web browser.
