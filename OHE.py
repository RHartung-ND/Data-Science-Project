import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from collections import Counter

df = pd.read_csv('movie-data/Imdb_Movie_Dataset.csv')

def encode_with_rare(df, column, top_n=5):
    counts = Counter()
    df[column].apply(lambda x: counts.update([k.strip() for k in str(x).split(',') if pd.notna(x)]))
    top_items = [item for item, count in counts.most_common(top_n)]
    mlb = MultiLabelBinarizer(classes=top_items)
    item_list = df[column].apply(lambda x: [item.strip() for item in str(x).split(',') if pd.notna(x)])
    encoded_items = mlb.fit_transform(item_list)
    df_encoded_items = pd.DataFrame(encoded_items, columns=mlb.classes_)
    df = pd.concat([df, df_encoded_items], axis=1)

    def has_rare_item(items):
        if pd.isna(items):
            return 0
        item_list = [i.strip() for i in items.split(',')]
        for item in item_list:
            if item not in top_items:
                return 1
        return 0

    df[f'rare_{column}'] = df[column].apply(has_rare_item)
    df = df.drop([column], axis=1)
    return df

# Apply to relevant columns
columns_to_encode = ['genres', 'spoken_languages', 'keywords', 'production_countries']
for col in columns_to_encode:
    df = encode_with_rare(df, col)

#One hot encode original language.
encoder_language = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_language = encoder_language.fit_transform(df[['original_language']])
language_columns = encoder_language.get_feature_names_out(['original_language'])
df_encoded_language = pd.DataFrame(encoded_language, columns=language_columns)
df = pd.concat([df, df_encoded_language], axis=1)
df = df.drop(['original_language'], axis=1)

# print(df)
df.to_csv("OHE.csv")
