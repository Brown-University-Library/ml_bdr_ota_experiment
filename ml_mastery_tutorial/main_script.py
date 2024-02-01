import sys

# mlp for multi-label classification
from numpy import asarray
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
# from keras.models import Sequential
# from keras.layers import Dense
from sklearn.metrics import accuracy_score

"""
This demonstrates a multi-label classification task using the make_multilabel_classification function from the sklearn.datasets module.

It generates a synthetic dataset with 1000 samples, 10 features, 3 classes, and 2 labels per sample.

The function prints the shape of the dataset (X and y) and the first few examples.
"""

# example of a multi-label classification task
from sklearn.datasets import make_multilabel_classification

def get_dataset():
	# define dataset
	X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)
	# X, y = make_multilabel_classification(n_samples=100, n_features=10, n_classes=3, n_labels=2, random_state=1)  # TEMP!!!
	# summarize dataset shape
	print(X.shape, y.shape)
	# summarize first few examples
	for i in range(10):
		print(X[i], y[i])
	return X, y
	
# def get_dataset():

def validate_dataset(X, y):
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
		model.fit(X_train, y_train, verbose=0, epochs=100)
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

# load dataset
X, y = get_dataset()
# validate dataset
if validate_dataset(X, y):
	print("Dataset is valid")
else:
	sys.exit("Dataset is invalid")
n_inputs, n_outputs = X.shape[1], y.shape[1]

sys.exit("Stopping for testing")

# get model
model = get_model(n_inputs, n_outputs)
# fit the model on all data
model.fit(X, y, verbose=0, epochs=100)
# make a prediction for new data
row = [3, 3, 6, 7, 8, 2, 11, 11, 1, 3]
newX = asarray([row])
yhat = model.predict(newX)
print('Predicted: %s' % yhat[0])