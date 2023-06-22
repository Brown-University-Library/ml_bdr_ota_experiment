##############################################
# Broad Themes:
# Agriculture
# Armed Forces
# Commerce
# Corruption
# Courts
# Diplomatic Relations
# Dissenters
# Economic Assistance
# Economic Policy
# Education
# Geopolitics
# Industrial Relations
# Investments
# Medical Care
# Press
# Public Administration
# Religion
# Social Conditions
##############################################

import pandas as pd
import json
import pprint

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from sklearn.model_selection import train_test_split


def load_data():
    # Get the docs from the raw json file
    raw_data_dict = json.load(open('source_data/OtA_raw.json'))
    # pprint.pprint(raw_data_dict[0])
    # load the docs into a dataframe
    df = pd.DataFrame(raw_data_dict)
    return df

def df_to_dataset(df, shuffle=True, batch_size=32):
    # Convert the dataframe into a tf.data.Dataset
    df = df.copy()
    labels = df.pop('mods_subject_broad_theme_ssim')
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

def train_val_test_split(df, test_size=0.1):
    print(df.shape)
    print(df.head())
    # split the data into train, validation, and test sets
    train, test = train_test_split(df, test_size=test_size)
    train, val = train_test_split(train, test_size=test_size)
    return train, val, test

def dfs_to_datasets(train_df, val_df, test_df, batch_size=64):
    # Convert the dataframes into tf.data.Datasets
    train_ds = df_to_dataset(train_df, batch_size=batch_size)
    val_ds = df_to_dataset(val_df, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test_df, shuffle=False, batch_size=batch_size)
    return train_ds, val_ds, test_ds

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    # Create a StringLookup layer which will turn strings into integer indices
    if dtype == 'string':
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        index = preprocessing.IntegerLookup(max_tokens=max_tokens)

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Create a Discretization for our integer indices.
    encoder = preprocessing.CategoryEncoding(num_tokens=index.vocabulary_size())

    # Apply one-hot encoding to our indices and return this feature
    return lambda feature: encoder(index(feature))



def encode_categorical_columns(all_inputs, encoded_features, categorical_columns):
    # Convert categorical columns to numeric
    for column in categorical_columns:
        categorical_column = tf.keras.Input(shape=(1,), name=column, dtype='string')
        encoding_layer = get_category_encoding_layer(column, train_ds, dtype='string', max_tokens=5)
        encoded_categorical_column = encoding_layer(categorical_column)
        all_inputs.append(categorical_column)
        encoded_features.append(encoded_categorical_column)
    return encoded_features

def manager():
    # load the data
    df = load_data()

    # create a copy of the dataframe with only the columns we want to use
    df = df[['pid','genre','keyword', 'mods_location_physical_location_ssim', 'mods_language_code_ssim', 'mods_subject_broad_theme_ssim']]


    # drop rows with no values in the mods_subject_broad_theme_ssim column
    df = df.dropna(subset=['mods_subject_broad_theme_ssim'])

    # split the data into train, validation, and test sets
    train, val, test = train_val_test_split(df)

    # Convert the dataframes into tf.data.Datasets
    train_ds, val_ds, test_ds = dfs_to_datasets(train, val, test)
    
    # Define the categorical columns
    categorical_columns = ['mods_location_physical_location_ssim', 'mods_language_code_ssim', 'mods_subject_broad_theme_ssim']

    # Encode the categorical columns
    all_inputs = []
    encoded_features = []
    encoded_features = encode_categorical_columns(all_inputs, encoded_features, categorical_columns)

    # Print the encoded features to see what they look like
    print('Encoded Features:')
    pprint.pprint(encoded_features)

if __name__ == '__main__':
    manager()