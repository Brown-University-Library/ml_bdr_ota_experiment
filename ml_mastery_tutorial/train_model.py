import sys

import numpy as np

from keras.layers import Dense
from keras.models import Sequential

from icecream import ic

def open_dataset():
    '''
    Open the dataset, stored as a npz file.

    Returns:
        X: The input data
        y: The output data
    '''
    # load the dataset
    data = np.load('dataset.npz')
    X, y = data['X'], data['y']
    ic(X.shape, y.shape)
    return X, y

def get_model(n_inputs, n_outputs):
    """ Creates and compiles a Sequential neural network model 
            with a ReLU-activated layer and a sigmoid output layer for classification.
        See the ml_mastery_tutorial README_ml_mastery.md` for great information.
        Called by both manage_original_dataset_processing() and manage_toy_dataset_processing() """
    # create a sequential model
    model = Sequential()
    # add a dense layer with 20 units, using 'relu' activation function
    model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    # add a dense layer with 10 units, using 'relu' activation function #TEST 
    model.add(Dense(10, kernel_initializer='he_uniform', activation='relu'))
    # add a dense layer with n_outputs units, using 'sigmoid' activation function
    model.add(Dense(n_outputs, activation='sigmoid'))
    # compile the model with binary cross-entropy loss and adam optimizer
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def manage_training(X, y):
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # print the shape of the dataset
    print(f'X.shape: {X.shape}, y.shape: {y.shape}')






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

    return model

if __name__ == '__main__':
    # open the dataset
    X, y = open_dataset()
    # manage the training process
    model = manage_training(X, y)
    # save the model
    model.save('model.keras')