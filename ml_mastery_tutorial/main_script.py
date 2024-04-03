## python
import sys

## third-party
import pandas
from keras.layers import Dense
from keras.models import Sequential
from numpy import asarray
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold

# # Global Variable in use!!
# # global_feature_columns


## original dataset helper functions --------------------------------


def get_original_dataset():
    """ Generates a synthetic dataset with 1000 samples, 10 features, 3 classes, and 2 labels per sample.
        The function prints the shape of the dataset (X and y) and the first few examples.
        Called by manage_original_dataset_processing() """
    # define dataset
    X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)  # type: ignore -- make_multilabel_classification() _can_ return a four-element tuple if the `return_indicator` parameter is True
    # summarize dataset shape
    print(X.shape, y.shape)
    # summarize first few examples
    print( '\nfirst few examples of original dataset...' )
    for i in range(10):
        print(X[i], y[i])  # type: ignore
    return X, y


def validate_original_dataset(X, y): # For the original dataset from tutorial
    """ Validates the original dataset.
        Called by manage_original_dataset_processing() """
    # validate that the dataset matches the expected shape:
    # example: [3 5 2 5 6 3 1 5 2 4] [1 0 1]
    try:
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        assert y.shape[1] == 3, "y must have 3 classes"
        assert X.shape[1] == 10, "X must have 10 features"
        assert y.shape[0] == 1000, "y must have 1000 samples"
        assert X.shape[0] == 1000, "X must have 1000 samples"
        # assert that the training datatypes are numeric
        assert X.dtype == 'float64', "X must be of type float64"
        # assert that the label datatypes are integers
        assert y.dtype == 'int64', "y must be of type int64"
        # assert that the label values are 0 or 1
        assert y.min() >= 0, "y must have a minimum value of 0"
        assert y.max() <= 1, "y must have a maximum value of 1"

    except AssertionError as e:
        print(e)
        return False
    return True


def evaluate_model(X, y):
    """ Evaluates model using repeated k-fold cross-validation
        Called by manage_original_dataset_processing() """
    results = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        #prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # define model
        model = get_model(n_inputs, n_outputs)
        # fit model
        print( 'running training model.fit()' )
        model.fit(X_train, y_train, verbose=0, epochs=100)  # type: ignore
        print( 'finished training model.fit()' )
        # make a prediction on the test set
        yhat = model.predict(X_test)
        # round probabilities to class labels
        yhat = yhat.round()
        # calculate accuracy
        acc = accuracy_score(y_test, yhat)
        # store result
        print('>%.3f' % acc)
        results.append(acc)
    return results


## end original dataset helper functions ----------------------------


## toy dataset helper functions -------------------------------------


def create_toy_dataset():
    """ Creates a toy-dataset, converts it to a pandas dataframe, and returns the dataframe.
        Called by get_dataset() """
    # This dataset will be composed of songs, with the following features:
    # - genre
    # - artist
    # - decade
    # The labels will be:
    # - has_guitar
    # - has_saxophone
    # - has_vocals

    # define dataset
    possible_genres = ['rock', 'pop', 'jazz', 'blues', 'country']
    possible_artists = ['Rocky', 'Popsicle', 'Jazzy Jeff', 'Blueman Group', 'Country Joe', 'Jazz on the Rocks', 'Blue Note Rock', 'Country of Pop']
    possible_decades = ['60s', '70s', '80s', '90s']

    X = []
    y = []
    for i in range(1000):
        # generate random features
        genre = possible_genres[i % len(possible_genres)]
        artist = possible_artists[i % len(possible_artists)]
        decade = possible_decades[i % len(possible_decades)]
        # generate random labels
        has_guitar = 1 if genre in ['rock', 'blues', 'country'] else 0
        has_saxophone = 1 if genre in ['jazz', 'blues'] else 0
        has_vocals = 1 if artist in ['Rocky', 'Popsicle', 'Jazzy Jeff', 'Country Joe'] else 0
        # store
        X.append([genre, artist, decade])
        y.append([has_guitar, has_saxophone, has_vocals])
    # convert to numpy array
    X = asarray(X)
    y = asarray(y)
    # convert to pandas dataframe
    df = pandas.DataFrame(X, columns=['genre', 'artist', 'decade'])
    # convert the values to strings
    df = df.astype('string')
    # print info about the dataframe
    print(df.info())

    # add the labels to the dataframe
    df['has_guitar'] = y[:, 0]
    df['has_saxophone'] = y[:, 1]
    df['has_vocals'] = y[:, 2]
    # summarize dataset shape
    print(X.shape, y.shape)
    # print info about the dataframe
    print(df.info())

    # Print the header row
    print("X: genre, artist, decade")
    print("y: has_guitar, has_saxophone, has_vocals")
    # summarize first few examples
    print( '\nfirst few examples of toy dataset...')
    print(df.head())
    # Save the dataframe to a csv file
    df.to_csv('toy_dataset.csv', index=False)
    return df

    # end def create_toy_dataset()


def one_hot_encode(df, column_name):
    """ One-hot encodes a column in a dataframe.
        For reference:
            One-hot encoding transforms categorical data, like colors, into a format 
            that machine learning models can understand—using binary vectors. Imagine we have a 
            dataset with the column "Color" that contains three records: "Red", "Blue", and "Green". 
            One-hot encoding would create three new columns, one for each color. For the record 
            with "Red", the "Red" column would have a "1", and the "Blue" and "Green" columns would 
            have "0"s. This binary representation allows algorithms to process the categorical data 
            without assuming an order or hierarchy among the colors. (chatgpt4)
        Called by get_dataset() """
    # one-hot encode the column
    one_hot = pandas.get_dummies(df[column_name], dtype='int64')
    print(f'one_hot for column {column_name}:\n{one_hot}')
    # drop the original column
    df = df.drop(column_name, axis=1)
    # join the new one-hot encoded columns
    df = df.join(one_hot)
    return df

# def one_hot_encode_test_row(test_row: dict) -> pandas.DataFrame:
#     temp_df = pandas.DataFrame(test_row, index=[0])
#     print(f'{temp_df = }')
#     # one-hot encode each column
#     for column_name in temp_df.columns:
#         temp_df = one_hot_encode(temp_df, column_name)

#     print(f'one-hot encoded temp_df:\n')
#     print(temp_df.head())

#     return temp_df

def one_hot_encode_test_row(test_row: dict) -> pandas.DataFrame:
    # Create a dataframe from the test_row dictionary using the global_feature_columns as the columns
    temp_df = pandas.DataFrame(columns=global_feature_columns)
    # Create a row with all False values
    temp_df.loc[0] = False
    # print(f'------\n{temp_df = }')
    # Iterate through the values in test_row and look for the corresponding column in temp_df
    for key, value in test_row.items():
        # print(f'{key = }, {value = }')
        if value in temp_df.columns:
            # print(f'{value = } is in temp_df.columns')
            # If the column exists, set the value in the first row to True
            temp_df.at[0, value] = True
    # print(f'After: \n{temp_df}')
    return temp_df


def get_dataset():
    """ Creates a toy-dataset in a dataframe.
        Then one-hot encodes the categorical features.
        Then converts the dataframe to back to numpy arrays.
        Called by manage_toy_dataset_processing() """
    df = create_toy_dataset()
    # one-hot encode the categorical features
    feature_columns = df.columns[:-3] # all columns except the last 3
    print(f'feature_columns: {feature_columns}')
    for column_name in feature_columns:
        df = one_hot_encode(df, column_name)
    # df = one_hot_encode(df, 'genre')
    print(f'one-hot encoded df:\n')
    print(df.head())

    updated_feature_columns = df.columns[3:] # all columns except the first 3 (first 3 are labels)
    print(f'updated_feature_columns: {updated_feature_columns}')

    # Assign updated_feature_columns to global variable
    global global_feature_columns
    global_feature_columns = updated_feature_columns

    # print info about the dataframe
    print('-'*40)
    print('df.info()')
    print(df.info())
    print('-'*40)

    # convert the dataframe to numpy arrays
    X = df[updated_feature_columns].values
    y = df[['has_guitar', 'has_saxophone', 'has_vocals']].values
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


def validate_dataset(X, y): # For the toy dataset created by us (music dataset)
    """ Validates the toy-dataset.
        Called by manage_toy_dataset_processing() """
    # validate that the dataset matches the expected shape:
    # example: ['rock' 'Rocky' '60s'] [1 0 1]
    try:
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        assert y.shape[1] == 3, "y must have 3 classes"
        assert X.shape[1] == 3, "X must have 3 features"
        assert y.shape[0] == 1000, "y must have 1000 samples"
        assert X.shape[0] == 1000, "X must have 1000 samples"
        # assert that the training data values are strings
        for i in range(X.shape[1]):
            for j in range(X.shape[0]):
                assert type(X[j][i]) == str, f'X[{j}][{i}] must be of type string, but is ```{type(X[j][i])}```'

        assert X.dtype == 'object', f'X must be of type object, but is ```{X.dtype}```'
        # assert that the label datatypes are integers
        assert y.dtype == 'int64', "y must be of type int64"
        # assert that the label values are 0 or 1
        assert y.min() >= 0, "y must have a minimum value of 0"
        assert y.max() <= 1, "y must have a maximum value of 1"

    except AssertionError as e:
        print(e)
        return False
    return True


## end toy dataset helper functions ---------------------------------


## common helper function(s) ----------------------------------------


def get_model(n_inputs, n_outputs):
    """ Creates and compiles a Sequential neural network model 
            with a ReLU-activated layer and a sigmoid output layer for classification.
        See the ml_mastery_tutorial README_ml_mastery.md` for great information.
        Called by both manage_original_dataset_processing() and manage_toy_dataset_processing() """
    # create a sequential model
    model = Sequential()
    # add a dense layer with 20 units, using 'relu' activation function
    model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    # add a dense layer with n_outputs units, using 'sigmoid' activation function
    model.add(Dense(n_outputs, activation='sigmoid'))
    # compile the model with binary cross-entropy loss and adam optimizer
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


## end common helper function(s) ------------------------------------




## manage original dataset processing -------------------------------
def manage_original_dataset_processing():
    """ Manages the original dataset processing.
        This demonstrates a multi-label classification task using the 
            make_multilabel_classification function from the sklearn.datasets module.
        Called by dundermain. """
    # load dataset
    X, y = get_original_dataset()
    # # validate dataset
    # if validate_dataset(X, y):
    #     print("Dataset is valid")
    # else:
    #     sys.exit("Dataset is invalid")

    if validate_original_dataset(X, y):
        print("Dataset is valid")
    else:
        sys.exit("Dataset is invalid")

    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # print the shape of the dataset
    print(f'X.shape: {X.shape}, y.shape: {y.shape}')

    # use one_hot_encode to encode the categorical features


    # sys.exit("Stopping for testing")

    # get model
    model = get_model(n_inputs, n_outputs)
    # fit the model on all data
    print( 'running manager model.fit()' )
    # model.fit(X, y, verbose=0, epochs=100)
    model.fit(X, y, verbose=0, epochs=100)  # type: ignore
    print( 'finished manager model.fit()' )


    # # evaluate the model
    results = evaluate_model(X, y)
    print( f'Accuracy: {std(results):.3f} ({results})' )

    ## end def manage_original_dataset_processing()


## manage toy-dataset processing ------------------------------------
def manage_toy_dataset_processing():
    """ Manages the toy dataset processing.
        Called by dundermain. """
    # load dataset
    X, y = get_dataset()
    # # validate dataset
    # if validate_dataset(X, y):
    #     print("Dataset is valid")
    # else:
    #     sys.exit("Dataset is invalid")

    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # print the shape of the dataset
    print(f'X.shape: {X.shape}, y.shape: {y.shape}')

    # use one_hot_encode to encode the categorical features


    # sys.exit("Stopping for testing")

    # get model
    model = get_model(n_inputs, n_outputs)
    # fit the model on all data
    print( 'running manager model.fit()' )
    # model.fit(X, y, verbose=0, epochs=100)
    model.fit(X, y, verbose=0, epochs=100)  # type: ignore
    print( 'finished manager model.fit()' )


    # # # evaluate the model
    # results = evaluate_model(X, y)
    # print( f'Standard Deviation: {std(results):.3f}  Accuracy Scores: ({results})' )
    # print( f'Averaged accuracy: {sum(results)/len(results):.3f}')

    # make a prediction for new data
    # row = [3, 3, 6, 7, 8, 2, 11, 11, 1, 3]
    '''
     ['blues', 'country', 'jazz', 'pop', 'rock', 'Blue Note Rock',
       'Blueman Group', 'Country Joe', 'Country of Pop', 'Jazz on the Rocks',
       'Jazzy Jeff', 'Popsicle', 'Rocky', '60s', '70s', '80s', '90s']
    '''

    # FOR NEXT TIME: Create process to allow testing row in a more sensible way (i.e. using the same one-hot encoding process as the training data)
    # BJD Has an idea involving a dictionary

    print(f'{global_feature_columns = }')
    # sys.exit("Stopping for testing")

    #FOR NEXT TIME: Revise one_hot_encode_test_row() to take into account all the columns (we're using a global variable to store the column names)

    # row = [True, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False]
    test_rows = [
                {'genre': 'blues', 'artist': 'Blue Note Rock', 'decade': '70s'},
                {'genre': 'country', 'artist': 'Country Joe', 'decade': '80s'},
                {'genre': 'jazz', 'artist': 'Jazz on the Rocks', 'decade': '90s'},
                {'genre': 'pop', 'artist': 'Country of Pop', 'decade': '60s'},
                {'genre': 'rock', 'artist': 'Popsicle', 'decade': '70s'}
                ]

    for test_row in test_rows:
        print(f'Test row: {test_row}')

        encoded_test_row = one_hot_encode_test_row(test_row=test_row)

        # newX = asarray([row])
        newX = encoded_test_row.values
        yhat = model.predict(newX)
        print('             has_guitar, has_saxophone, has_vocals')
        print(f'Predicted:    ',end='')
        for i in range(yhat.shape[1]):
            print(f'{yhat[0][i]:.2f}', end='        ')
        print()
          
    ## end of manage_toy_dataset_processing()
    

## ------------------------------------------------------------------
if __name__ == "__main__":
    """
    Possible TODOs: 
    - add command-line argument to specify which dataset to process
    - add `manage_small_real_dataset_processing()`
    - add `manage_full_real_dataset_processing()`
    """
    # manage_original_dataset_processing()
    manage_toy_dataset_processing()
