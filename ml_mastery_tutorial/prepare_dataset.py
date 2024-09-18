from hashlib import md5

import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
# from scipy import sparse
from scipy.sparse import spmatrix

from icecream import ic


def read_in_dataset(dataset_path: str | None = None) -> pd.DataFrame:
    """
    Reads in the real data from a json file and 
    returns a pandas dataframe.
    """
    if dataset_path is None:
        raise ValueError("You must provide a dataset path")
    df = pd.read_json(dataset_path)
    return df

def filter_empty(x):
    '''
    This function is used to filter out rows that have empty values in the
    mods_subject_broad_theme_ssim column.

    Args:
        x: The value to check.

    Returns:
        True if the value is not empty, False otherwise.
    '''
    # ic(x)
    # ic(type(x))
    # ic(len(x))
    # ic(type(x[0]))
    length = len(x)
    if length < 2:
        # print('|||')
        first_element_length = len(x[0])
        # ic(first_element_length)
        if first_element_length < 1:
            # print('Will be removed')
            # sys.exit("Found One!!")
            return False
    return True

def pick_most_common_values(df: pd.DataFrame, column_name: str, n_values: int
                            ) -> dict[str, int]:
    """
    Build a dictionary of the most common values in a column. Values will be
    lists, even when there is only one value. We want the values from within
    the lists, not the lists themselves.
    

    Args:
        df: The dataframe to process.
        column_name: The name of the column to process.
        n_values: The number of most common values to keep.

    Returns:
        The dataframe with the column values pruned to the top n_values.

    Ultimate Goal: if the row-1 value is ['a','b','c'] and the row-2 value is 
    ['c','d','e'] -- and we determine that 'c' doesn't meet the 
    threshold-count we determine, then row-1 would be re-written 
    to: ['a','b'] and row-2 would be re-written to: ['d','e'].
    """
    value_counts: dict[str, int] = {}

    # Iterate through the rows in the column
    for index, row in df.iterrows():
        # If the value is a list, iterate through the list
        if isinstance(row[column_name], list):
            for value in row[column_name]:
                if value in value_counts:
                    value_counts[value] += 1
                else:
                    value_counts[value] = 1
        # If the value is not a list, add it to the counts
        else:
            if row[column_name] in value_counts:
                value_counts[row[column_name]] += 1
            else:
                value_counts[row[column_name]] = 1

    # Sort the values by count
    sorted_values = {k: v for k, v in 
                     sorted(value_counts.items(), 
                            key=lambda item: item[1], reverse=True)}
    
    # Keep only the top n_values
    top_values = {k: sorted_values[k] for k in list(
        sorted_values.keys())[:n_values]}

    # # Print debug information
    # ic(len(value_counts))
    # ic(f'{top_values = }')

    # # Iterate through the rows in the column
    # for index, row in df.iterrows():
    #     # If the value is a list, iterate through the list
    #     if isinstance(row[column_name], list):
    #         for value in row[column_name]:
    #             if value not in top_values:
    #                 row[column_name].remove(value)
    #     # If the value is not a list, check if it's in the top_values
    #     else:
    #         if row[column_name] not in top_values:
    #             row[column_name] = ''

    # # Print debug information
    # print(f'{df[column_name].value_counts()}')

    return top_values

def apply_common_value_filter(row: pd.Series, column_name: str, 
                            common_values: dict[str, int]) -> pd.Series:
    """
    Filters out values in a row that are not in the common_values dictionary.

    Args:
        row: The row to process.
        column_name: The name of the column to process.

    Returns:
        The row with the column values pruned to the common_values.
    """
    # We expect the values to be lists, 
    # so we'll convert them to lists if they're not
    if not isinstance(row[column_name], list):
        row[column_name] = [row[column_name]]

    # Filter out values that are not in the common_values dictionary
    row[column_name] = [value for value in row[column_name]
                        if value in common_values]
    
    return row

def one_hot_encode(df: pd.DataFrame, column_name: str
                   ) -> pd.DataFrame:
    """ 
    One-hot encodes a column with potential list values in a dataframe.
    This version handles both scalar and list values by ensuring all data is 
    treated as lists.
    """
    # Ensure all data in the column are lists
    df[column_name] = df[column_name].apply(lambda x: x if isinstance(x, list) else [x])

    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()

    # Fit the MultiLabelBinarizer and transform the data
    binarized_data: np.ndarray | spmatrix = mlb.fit_transform(df[column_name])
    ic(binarized_data)

    # Create a new dataframe with the one-hot encoded data
    # TODO: Investigate converting to a dense array or handling sparse arrays
    one_hot = pd.DataFrame(binarized_data, columns=mlb.classes_
                           , dtype='int64'
                           ) 

    # Print debug information
    print(f'one_hot for column {column_name}:\n{one_hot}')

    # Drop the original column
    df = df.drop(column_name, axis=1)

    # Join the new one-hot encoded columns
    df = df.join(one_hot, rsuffix=f'_{column_name}')
    
    return df


def get_dataset():
    """ 
    Converts a dataset from a csv file to a dataframe.
    Then one-hot encodes the categorical features.
    Then converts the dataframe to back to numpy arrays.
    """
    # df = create_toy_dataset()
    df = read_in_dataset('../source_data/OtA_raw.json')
    print(f'df.head():\n{df.head()}')

    # Remove columns that are not needed
    '''
    Keeping only:
    'pid','genre','keyword', 
    'mods_location_physical_location_ssim', 
    'mods_language_code_ssim', 'mods_subject_broad_theme_ssim'
    '''
    df = df[['pid', 'genre', 'keyword', 'mods_location_physical_location_ssim', 'mods_language_code_ssim', 'mods_subject_broad_theme_ssim']]
    print(f'df.head() after removing columns:\n{df.head()}')
    print(f'Columns in df:\n{df.columns}')

    ###
    # Remember to deal with the pid column better later
    ###

    # Convert NaN values to ['']
    df = df.fillna('')
    df = df.map(lambda x: x if isinstance(x, list) else [x])

    # Save the dataframe to a csv file
    df.to_csv('cleaned_dataset.csv', index=False)

    # Remove rows with empty values in the 'mods_subject_broad_theme_ssim' column
    df = df[df['mods_subject_broad_theme_ssim'].map(filter_empty)]

    # Reset the index
    '''
    We needed to do this so that the dataframes returned by one_hot_encode
    would have the same index as the original dataframe. Without doing this,
    we found that the final dataframe had a large number of rows at the end
    that were full of NaN values. This also allows the final values to remain
    ints instead of being converted to floats.
    '''
    df = df.reset_index(drop=True)

    ## one-hot encode the categorical features

    # Get the feature columns
    feature_columns = df.columns[1:-1] # all columns except the label(s) and pids
    print(f'feature_columns: {feature_columns}')

    # Print all the unique values in 'genre', after temporarily converting the values to strings
    print(f'Unique values in genre: {df["genre"].astype(str).unique()}')

    ic('Before top_values:',df['keyword'])

    # Get the top n most common values in the 'keyword' column
    top_keyword_values = pick_most_common_values(df, 'keyword', 1000)

    # Apply the common value filter to the 'keyword' column
    df = df.apply(
        lambda x: apply_common_value_filter(x, 'keyword', 
                                            top_keyword_values), axis=1)
    
    ic('After top_values:',df['keyword'])


    # sys.exit("Stopping for testing")

    ic(feature_columns)
    for column_name in feature_columns:
        print(f'{column_name = }')
        df = one_hot_encode(df, column_name)
    # df = one_hot_encode(df, 'genre')
    print(f'one-hot encoded df:\n')
    print(df.head())

    updated_feature_columns = df.columns[1:] # all columns except the first (first is label)
    print(f'updated_feature_columns: {updated_feature_columns}')

    # Print all columns with the word 'keyword' in them
    print('Columns with "keyword" in them: '
          f'{df.columns[df.columns.str.contains("keyword")]}')

    # Split out the mods_subject_broad_theme_ssim column into a separate dataframe
    df_labels = pd.DataFrame(df['mods_subject_broad_theme_ssim'])

    # Drop the mods_subject_broad_theme_ssim column from the main dataframe
    df = df.drop('mods_subject_broad_theme_ssim', axis=1)

    # One-hot encode the mods_subject_broad_theme_ssim column
    df_labels = one_hot_encode(df_labels, 'mods_subject_broad_theme_ssim')
    

    ic(df)
    ic(df_labels)

    # # Print the first row in detail of both dataframes
    # for column_name in df.columns[:50]:
    #     print(f'{column_name}:  {df.iloc[0][column_name]}')
    #     print(f'{column_name}:  {df.iloc[20129][column_name]}')



    '''
    !!!!!!!!!!!!!!!!
    MARK: Stopping Here

    We discovered that a large number of rows at the end of the df are full
    of NaN values. We need to figure out why
        ** Done **

    1. Need to determine which of the other columns to filter (if any)
        ** Come back to this later **
    2. Look into how columns are being joined to the dataframe (make
       sure duplicate names are not being merged...including empty values)
       ** Done **
    3. Determine what we've done and still need to do with the 
        mods_subject_broad_theme_ssim column
        a. We need to remove the rows that have empty values for training
            ** Done **
        b. We need to one-hot encode the values
            ** Done sort of ** See above
        c. Other???
    4. Figure out how to procede with real data    
    !!!!!!!!!!!!!!!!
    '''
    # Assign updated_feature_columns to global variable
    global global_feature_columns
    global_feature_columns = updated_feature_columns

    # print info about the dataframe
    print('-'*40)
    print('df.info()')
    print(df.info())
    print('-'*40)

    # # convert the dataframe to numpy arrays
    # X = df[updated_feature_columns].values
    # y = df[['has_guitar', 'has_saxophone', 'has_vocals']].values
    # # print X for debugging
    # print('   X   ')
    # print('-'*40)
    # print(X)
    # print('-'*40)

    # Remove mods_subject_broad_theme_ssim from the updated_feature_columns
    updated_feature_columns = updated_feature_columns[1:]
    ic(updated_feature_columns)

    # convert the dataframe to numpy arrays
    X = df[updated_feature_columns].values
    y = df_labels.values

    # print X for debugging
    print('   X   ')
    print('-'*40)
    print(X)
    print('-'*40)

    # print y for debugging
    print('   y   ')
    print('-'*40)
    print(y)
    print('-'*40)

    return X, y


    ## end def get_dataset()

def manage_dataset_prep():
    """ Manages the dataset processing.
        Called by dundermain. """
    # load dataset
    X, y = get_dataset()

    # confirm dataset is cleaned the same over multiple runs
    X_bytes = X.tobytes()
    assert md5(X_bytes).hexdigest() == 'c242519776e782520a61563cdd63a214', \
        f'X hash is {md5(X_bytes).hexdigest()}'
    y_bytes = y.tobytes()
    assert md5(y_bytes).hexdigest() == 'd8c459c8dcd0625b555e71db7a992309', \
        f'y hash is {md5(y_bytes).hexdigest()}'

    # # validate dataset
    # if validate_dataset(X, y):
    #     print("Dataset is valid")
    # else:
    #     sys.exit("Dataset is invalid")

    # print the shape of the dataset
    print(f'X.shape: {X.shape}, y.shape: {y.shape}')

    # Save the dataset to a file
    np.savez_compressed('dataset.npz', X=X, y=y)

    return X, y

if __name__ == '__main__':
    manage_dataset_prep()






