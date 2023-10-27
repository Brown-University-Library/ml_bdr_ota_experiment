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

import logging
import sys
import pandas as pd
import json
import pprint

import tensorflow as tf

from tensorflow.keras import layers  
from tensorflow.keras.layers.experimental import preprocessing

try:
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:  # happens on macOS in py3.8 venv
    from scikit_learn.model_selection import train_test_split


## create basicConfig logger with no file-writer
logging.basicConfig(
    level=logging.DEBUG,
    # format='%(asctime)s %(levelname)s %(message)s',
    format='[%(asctime)s] %(levelname)s [%(module)s-%(funcName)s()::%(lineno)d] %(message)s',
    datefmt='%d/%b/%Y %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)
log.debug( 'test log entry' )



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
    # log.debug( f'labels, ``{pprint.pformat(labels)}``')
    # print(f'-=-=-=labels: {labels}')
    # print(f'type(labels): {type(labels)}')

    # print('=-=-=-=-=-=-=-=-=-=-=-=')
    # print(f'The df dict is: {pprint.pformat(dict(df))[:1000]}')
    # print('=-=-=-=-=-=-=-=-=-=-=-=')

    # log.debug( 'dictifying df' )
    the_dict = dict(df)
    # log.debug( f'about to process the_dict, ``{pprint.pformat(the_dict)}``' )
    item = list(the_dict.items())[0]
    # log.debug( f'item, ``{item}``' )

    # log.debug( f'the value type, ``{type(item[1])}``' )
    # log.debug( 'about to instantiate ds' )
    ds = tf.data.Dataset.from_tensor_slices((the_dict, labels))
    # log.debug( 'ds instantiated' )

    # print('made it past ds=...')
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

def train_val_test_split(df, test_size=0.1):
    # print(df.shape)
    # print(df.head())
    # split the data into train, validation, and test sets
    train, test = train_test_split(df, test_size=test_size)
    train, val = train_test_split(train, test_size=test_size)
    # print(f'train type: {type(train)}')
    # print(f'train shape: {train.shape}')
    # print(f'train head: {train.head()}')
    # print(f'val type: {type(val)}')
    # print(f'val shape: {val.shape}')
    # print(f'val head: {val.head()}')
    # print(f'test type: {type(test)}')
    # print(f'test shape: {test.shape}')
    # print(f'test head: {test.head()}')
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

def get_category_encoding_layer_label(dataset, dtype, max_tokens=None):
    # Create a StringLookup layer which will turn strings into integer indices
    if dtype == 'string':
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        index = preprocessing.IntegerLookup(max_tokens=max_tokens)
            
    # Prepare a Dataset that only yields the label
    feature_ds = dataset.map(lambda x, y: y)

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Create a Discretization for our integer indices.
    encoder = preprocessing.CategoryEncoding(num_tokens=index.vocabulary_size())

    # Apply one-hot encoding to our indices and return this feature
    return lambda label: encoder(index(label))



def encode_categorical_columns(all_inputs, dataset, encoded_features, categorical_columns):
    # Convert categorical columns to numeric
    for column in categorical_columns:
        categorical_column = tf.keras.Input(shape=(1,), name=column, dtype='string')
        # encoding_layer = get_category_encoding_layer(column, dataset, dtype='string', max_tokens=5)
        encoding_layer = get_category_encoding_layer(column, dataset, dtype='string')
        encoded_categorical_column = encoding_layer(categorical_column)
        all_inputs.append(categorical_column)
        encoded_features.append(encoded_categorical_column)
    return encoded_features


def stringify_list(passed_list, sort=False):
    """ 
    Converts submitted list to a string, removing whitespace and punctuation.
    If the submitted object is not a list, returns the object as a string.
    Usage:
    >>> stringify_list( ['a', 'b', 'c'] )
    'a_b_c'
    >>> stringify_list( ['a', 'b c', 'd'] )
    'a_b-c_d'
    >>> stringify_list( 'foo' )
    'foo'
    >>> stringify_list( 42 )
    '42'
    >>> stringify_list( ['b', 'a', 'd'], sort=True )
    'a_b_d'
    """
    # If the 'list' is not a list, return it as a string
    if type(passed_list) != list:
        return str(passed_list)
    # Convert a list to a string, removing whitespace and punctuation
    if sort:
        passed_list = sorted(passed_list)
    string = '_'.join(passed_list)
    string = string.replace(' ', '-')
    return string

# #compile
# all_features = tf.keras.layers.concatenate(encoded_features)
# x = tf.keras.layers.Dense(128, activation="relu")(all_features)
# x = tf.keras.layers.Dense(64, activation="relu")(x)
# x = tf.keras.layers.Dropout(0.1)(x)
# output = tf.keras.layers.Dense(1)(x)

# model = tf.keras.Model(all_inputs, output)

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=["accuracy"])

def compile_model(all_inputs, encoded_features, categorical_label_column, num_classes):
    all_features = tf.keras.layers.concatenate(encoded_features)
    x = tf.keras.layers.Dense(128, activation="relu")(all_features)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax', name=categorical_label_column)(x)

    model = tf.keras.Model(all_inputs, output)

    model.compile(optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=["accuracy"])
    log.debug( 'model compiled' )
    return model

def graph_model(model):
    log.debug( 'about to graph model' )
    # rankdir='LR' is used to make the graph horizontal.
    tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

def train_model(model, train_ds, val_ds, test_ds, num_classes, epochs=10):
    log.debug( 'about to train model' )
    exit=False
    # Check if the target labels are already one-hot encoded
    train_labels = train_ds.map(lambda x, y: y)
    if train_labels.element_spec.shape[-1] == num_classes:
        log.debug('Labels are already one-hot encoded')
    else:
        log.debug('Train Labels are not one-hot encoded')
        exit=True

    val_labels = val_ds.map(lambda x, y: y)
    if val_labels.element_spec.shape[-1] == num_classes:
        log.debug('Labels are already one-hot encoded')
    else:
        log.debug('Val Labels are not one-hot encoded')
        exit=True

    test_labels = test_ds.map(lambda x, y: y)
    if test_labels.element_spec.shape[-1] == num_classes:
        log.debug('Labels are already one-hot encoded')
    else:
        log.debug('Test Labels are not one-hot encoded')
        exit=True

    if exit:
        raise Exception('Labels are not one-hot encoded')

    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    log.debug( 'training complete' )
    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)
    print("Loss", loss)
    return model
    



## -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-



## -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-



def manager():
    '''
    load the data which was fetched with data_fetch.py
    Create a dataframe from the data
    '''
    df = load_data()

    '''
    create a copy of the dataframe with only the columns we want to use
    This is a small subset of the columns in the original dataframe
    After getting things to work with this subset, we can add more columns
    '''
    df = df[['pid','genre','keyword', 'mods_location_physical_location_ssim', 'mods_language_code_ssim', 'mods_subject_broad_theme_ssim']]


    '''
    Drop rows with no values in the mods_subject_broad_theme_ssim column
    This is because that is the column we are teaching the model to predict
    So we remove any rows that don't have a value in that column, because they can't be used to train the model
    '''
    df = df.dropna(subset=['mods_subject_broad_theme_ssim'])

    assert type(df['mods_subject_broad_theme_ssim'].iloc[0]) == list, type(df['mods_subject_broad_theme_ssim'].iloc[0])

    assert type(df['mods_subject_broad_theme_ssim'].iloc[73]) == list, type(df['mods_subject_broad_theme_ssim'].iloc[73])

    '''
    The next few steps are to convert the values in the dataframe into the format we want to use for training the model
    Ultimately, the values in the columns need to be converted into numeric values
    An intermediate step is to convert lists into strings and remove whitespace and punctuation
    '''

    # Convert the values in the mods_subject_broad_theme_ssim column to strings, removing whitespace and punctuation
    df['mods_subject_broad_theme_ssim'] = df['mods_subject_broad_theme_ssim'].apply(stringify_list, sort=True)

    assert type(df['mods_subject_broad_theme_ssim'].iloc[73]) == str, type(df['mods_subject_broad_theme_ssim'].iloc[73])

    # # Print the values in the mods_subject_broad_theme_ssim column to see what they look like
    # print('/\\'*50)
    # print('mods_subject_broad_theme_ssim:')
    # pprint.pprint(df['mods_subject_broad_theme_ssim'])
    # print('/\\'*50)

    # Get number of unique values in mods_subject_broad_theme_ssim column
    num_unique_themes = len(df['mods_subject_broad_theme_ssim'].unique())
    # print('/\\'*50)
    # print(f'Number of unique themes: {num_unique_themes}')
    # print('/\\'*50)

    # Stringify the other columns with list values
    df['genre'] = df['genre'].apply(stringify_list, sort=True)
    df['mods_language_code_ssim'] = df['mods_language_code_ssim'].apply(stringify_list, sort=True)
    df['mods_location_physical_location_ssim'] = df['mods_location_physical_location_ssim'].apply(stringify_list, sort=True)
    df['keyword'] = df['keyword'].apply(stringify_list, sort=True)


    # # Convert all values in the dataframe to strings
    # df = df.astype(str)

    print(f'Values 72-74 after stringifying:\n{df["mods_subject_broad_theme_ssim"].iloc[72:75]}')
    assert type(df['mods_subject_broad_theme_ssim'].iloc[73]) == str, type(df['mods_subject_broad_theme_ssim'].iloc[73])

    # Print the entire row corresponding to index 39084, for debugging
    print(f'Row 39084:\n{df.loc[39084]}')
    
    '''
    At this point we have a dataframe with values that are either numeric, or can be encoded as numeric (which will be done later)
    The next step is to split the data into train, validation, and test sets
    This is so we can train the model a subset of the data, and then test it on the rest of the data (which it has not seen before)
    The datasets are used in the following ways:
        training set: used to train the model. The model will be evaluated on this data during training,
            but it will not be used to evaluate the final model
        validation set: used to evaluate the model during training. The model will not be trained on this data, only evaluated on it.
            This is used to determine when the model has finished training. The model will not be trained further once it has reached,
            the point where it is no longer improving on the validation set.
        test set: used to evaluate the final model. The model will not be trained on this data, only evaluated on it.
            This is the final evaluation of the model, and the results are used to determine how well the model performs.
    '''

    # split the data into train, validation, and test sets
    try:
        train, val, test = train_val_test_split(df)
    except Exception as e:
        message = f'Unable to complete train_val_test_split: {e}'
        raise Exception(message)

    '''
    We convert the dataframes into tf.data.Datasets so they can be used to train the model with TensorFlow
    '''

    # Convert the dataframes into tf.data.Datasets
    try:
        train_ds, val_ds, test_ds = dfs_to_datasets(train, val, test)
    except Exception as e:
        message = f'Unable to complete dfs_to_datasets: {e}'
        raise Exception(message)
    
    '''
    Now we need to convert the categorical columns into numeric values by encoding them as one-hot vectors
    This means that each value in the column will be broken out into its own column,
    and the values will be represented by a 1 in that column if the value is present, and a 0 if it is not present
    This allows the model to operate on the values as numbers, rather than strings
    '''

    # Define the categorical columns
    categorical_columns = ['mods_location_physical_location_ssim', 'mods_language_code_ssim', 'genre', 'keyword']

    # Encode the categorical columns
    all_inputs = []
    encoded_features = []

    for column_name in categorical_columns:
        log.debug( f'column_name, ``{column_name}``' )
        categorical_column = tf.keras.Input(shape=(1,), name=column_name, dtype='string')
        # encoding_layer = get_category_encoding_layer(column_name, train_ds, dtype='string', max_tokens=5)
        encoding_layer = get_category_encoding_layer(column_name, train_ds, dtype='string')
        encoded_categorical_column = encoding_layer(categorical_column)
        all_inputs.append(categorical_column)
        encoded_features.append(encoded_categorical_column)
        log.debug( f'Done with column_name, ``{column_name}``' )

    # Print the encoded features to see what they look like
    print('='*50)
    print('Encoded Features:')
    pprint.pprint(encoded_features)

    log.debug( 'Encoding label' )
    categorical_label_column = tf.keras.Input(shape=(1,), name='mods_subject_broad_theme_ssim', dtype='string')
    # encoding_layer = get_category_encoding_layer_label(train_ds, dtype='string', max_tokens=5)
    encoding_layer = get_category_encoding_layer_label(train_ds, dtype='string')
    encoded_categorical_label_column = encoding_layer(categorical_label_column)

    print('|'*50)
    pprint.pprint(categorical_label_column)

    # Print the encoded label to see what it looks like
    print('-'*50)
    print('Encoded Label:')
    pprint.pprint(encoded_categorical_label_column)

    import numpy

    print('|-'*50)
    print('About to print out batches')

    # Run a small batch of data through the encoding layer to get the encoded label
    for batch in train_ds.take(1):
        print('+'*15)
        # print(batch)


        labels = batch[-1]
        encoded_labels = encoding_layer(labels)
        print("Encoded Labels:", encoded_labels.numpy())
        print('='*50)
        print('First encoded label:')
        print(encoded_labels[0].numpy())
        print('Second encoded label:')
        print(encoded_labels[1].numpy())

        print("Shape of Encoded Labels:", encoded_labels.shape)
        print("Data Type of Encoded Labels:", encoded_labels.dtype)


        # labels = batch['mods_subject_broad_theme_ssim']
        # encoded_labels = encoding_layer(labels)
        # print('='*50)
        # print('First encoded label:')
        # print(encoded_labels[0].numpy())  # Convert the first tensor in the batch to a NumPy array and print it



    # # Inspect the first encoded label
    # print('='*50)
    # print('First encoded label:')
    # # Need to find the right syntax for this


    # Compile the model
    log.debug( 'about to compile model' )
    model = compile_model(all_inputs, encoded_features, encoded_categorical_label_column, num_classes=num_unique_themes)

    # Graph the model
    log.debug( f'about to graph model' )
    graph_model(model)

    # Train the model
    log.debug( 'about to train model' )
    trained_model = train_model(model, train_ds, val_ds, test_ds, epochs=10, num_classes=num_unique_themes)



if __name__ == '__main__':
    manager()