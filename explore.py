# Imports
import pandas as pd
from plotly.express import histogram


### EXPLORE DATA ###
df = pd.read_csv(filepath_or_buffer='amz_us_price_prediction_dataset.csv',
                 index_col=['uid'])
# df = df.copy()
df = df[["reviews", "stars", "isBestSeller",
         "boughtInLastMonth", "price"]].copy()
df['has_reviews'] = df['reviews'] > 0
df['taxe_free_income'] = df['price'] * df['boughtInLastMonth']

# Pas concluant
# df['stars_x_reviews'] = df['stars'] * df['reviews']

corr_matrix = df.corr()
print(corr_matrix["price"])
print(corr_matrix["taxe_free_income"])

# print(df.info())
# print(df.head())
# print(df.describe())

histogram(data_frame=df, x='stars')
# histogram(data_frame=df, x='stars', y='taxe_free_income',
#           histfunc='avg', color='has_reviews')
# histogram(data_frame=df, x='stars', y='reviews',
#           histfunc='avg', color='has_reviews')
# histogram(data_frame=df, x='stars', color='has_reviews')
# histogram(data_frame=df, x='price', log_y=True)
# histogram(data_frame=df, x='price',
#           y='boughtInLastMonth', color='isBestSeller')
# histogram(data_frame=df, x='taxe_free_income',
#           y='price', histfunc='avg', log_y=True)
