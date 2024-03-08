import sys

# mlp for multi-label classification
from numpy import asarray
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
import pandas

"""
This demonstrates a multi-label classification task using the make_multilabel_classification function from the sklearn.datasets module.

It generates a synthetic dataset with 1000 samples, 10 features, 3 classes, and 2 labels per sample.

The function prints the shape of the dataset (X and y) and the first few examples.
"""

# example of a multi-label classification task
from sklearn.datasets import make_multilabel_classification

# def get_dataset():
# 	# define dataset
# 	X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)
# 	# X, y = make_multilabel_classification(n_samples=100, n_features=10, n_classes=3, n_labels=2, random_state=1)  # TEMP!!!
# 	# summarize dataset shape
# 	print(X.shape, y.shape)
# 	# summarize first few examples
# 	for i in range(10):
# 		print(X[i], y[i])
# 	return X, y

# def create_toy_dataset():
# 	# This dataset will be composed of songs, with the following features:
# 	# - genre
# 	# - artist
# 	# - decade
# 	# The labels will be:
# 	# - has_guitar
# 	# - has_saxophone
# 	# - has_vocals

# 	# define dataset
# 	possible_genres = ['rock', 'pop', 'jazz', 'blues', 'country']
# 	possible_artists = ['Rocky', 'Popsicle', 'Jazzy Jeff', 'Blueman Group', 'Country Joe', 'Jazz on the Rocks', 'Blue Note Rock', 'Country of Pop']
# 	possible_decades = ['60s', '70s', '80s', '90s']

# 	X = []
# 	y = []
# 	for i in range(1000):
# 		# generate random features
# 		genre = possible_genres[i % len(possible_genres)]
# 		artist = possible_artists[i % len(possible_artists)]
# 		decade = possible_decades[i % len(possible_decades)]
# 		# generate random labels
# 		has_guitar = 1 if genre in ['rock', 'blues', 'country'] else 0
# 		has_saxophone = 1 if genre in ['jazz', 'blues'] else 0
# 		has_vocals = 1 if artist in ['Rocky', 'Popsicle', 'Jazzy Jeff', 'Country Joe'] else 0
# 		# store
# 		X.append([genre, artist, decade])
# 		y.append([has_guitar, has_saxophone, has_vocals])
# 	# convert to numpy array
# 	X = asarray(X)
# 	y = asarray(y)
# 	# summarize dataset shape
# 	print(X.shape, y.shape)
# 	# Print the header row
# 	print("X: genre, artist, decade")
# 	print("y: has_guitar, has_saxophone, has_vocals")
# 	print('-'*40)
# 	print(X)
# 	print('-'*40)

# 	# summarize first few examples
# 	for i in range(10):
# 		print(X[i], y[i])
# 	return X, y

# create a toy dataset, convert to pandas dataframe, return the dataframe
def create_toy_dataset():
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
	print(df.head())
	return df

def one_hot_encode(df, column_name):
	# one-hot encode the column
	one_hot = pandas.get_dummies(df[column_name], dtype='int64')
	print(f'one_hot for column {column_name}:\n{one_hot}')
	# drop the original column
	df = df.drop(column_name, axis=1)
	# join the new one-hot encoded columns
	df = df.join(one_hot)
	return df

# STOPPED HERE: NEXT: do one-hot encoding of the categorical features, write a separate validation function for after the one-hot encoding


def get_dataset():
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

def validate_dataset(X, y): # For the toy dataset created by us (music dataset)
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



# def validate_dataset(X, y): # For the original dataset from tutorial
# 	# validate that the dataset matches the expected shape:
# 	# example: [3 5 2 5 6 3 1 5 2 4] [1 0 1]
# 	try:
# 		assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
# 		assert y.shape[1] == 3, "y must have 3 classes"
# 		assert X.shape[1] == 10, "X must have 10 features"
# 		assert y.shape[0] == 1000, "y must have 1000 samples"
# 		assert X.shape[0] == 1000, "X must have 1000 samples"
# 		# assert that the training datatypes are numeric
# 		assert X.dtype == 'float64', "X must be of type float64"
# 		# assert that the label datatypes are integers
# 		assert y.dtype == 'int64', "y must be of type int64"
# 		# assert that the label values are 0 or 1
# 		assert y.min() >= 0, "y must have a minimum value of 0"
# 		assert y.max() <= 1, "y must have a maximum value of 1"

# 	except AssertionError as e:
# 		print(e)
# 		return False
# 	return True

# get the model
# define a function to get the model
def get_model(n_inputs, n_outputs):
	# create a sequential model
	model = Sequential()
	# add a dense layer with 20 units, using 'relu' activation function
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	# add a dense layer with n_outputs units, using 'sigmoid' activation function
	model.add(Dense(n_outputs, activation='sigmoid'))
	# compile the model with binary cross-entropy loss and adam optimizer
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
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
		model.fit(X_train, y_train, verbose='0', epochs=100)
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

if __name__ == "__main__":

	# load dataset
	X, y = get_dataset()
	# # validate dataset
	# if validate_dataset(X, y):
	# 	print("Dataset is valid")
	# else:
	# 	sys.exit("Dataset is invalid")

	n_inputs, n_outputs = X.shape[1], y.shape[1]
	# print the shape of the dataset
	print(f'X.shape: {X.shape}, y.shape: {y.shape}')

	# use one_hot_encode to encode the categorical features


	# sys.exit("Stopping for testing")

	# get model
	model = get_model(n_inputs, n_outputs)
	# fit the model on all data
	model.fit(X, y, verbose='0', epochs=100)


	# # evaluate the model
	# results = evaluate_model(X, y)


	# make a prediction for new data
	# row = [3, 3, 6, 7, 8, 2, 11, 11, 1, 3]
	'''
	 ['blues', 'country', 'jazz', 'pop', 'rock', 'Blue Note Rock',
       'Blueman Group', 'Country Joe', 'Country of Pop', 'Jazz on the Rocks',
       'Jazzy Jeff', 'Popsicle', 'Rocky', '60s', '70s', '80s', '90s']
	'''

	# FOR NEXT TIME: Create process to allow testing row in a more sensible way (i.e. using the same one-hot encoding process as the training data)
	# BJD Has an idea involving a dictionary

	row = [True, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False]
	print(f'Test row: blues, Blue Note Rock, 70s')
	newX = asarray([row])
	yhat = model.predict(newX)
	print('has_guitar, has_saxophone, has_vocals')
	print('Predicted: %s' % yhat[0])