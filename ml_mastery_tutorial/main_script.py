# mlp for multi-label classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

"""
This demonstrates a multi-label classification task using the make_multilabel_classification function from the sklearn.datasets module.

It generates a synthetic dataset with 1000 samples, 10 features, 3 classes, and 2 labels per sample.

The function prints the shape of the dataset (X and y) and the first few examples.
"""

# example of a multi-label classification task
from sklearn.datasets import make_multilabel_classification
# define dataset
# X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)
X, y = make_multilabel_classification(n_samples=100, n_features=10, n_classes=3, n_labels=2, random_state=1)  # TEMP!!!
# summarize dataset shape
print(X.shape, y.shape)
# summarize first few examples
for i in range(10):
	print(X[i], y[i])
	


# define the model
model = Sequential()
model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')