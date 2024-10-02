import os
from typing import Union, Any

import keras
from keras.models import load_model
from icecream import ic

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the keras model from disk
def load_keras_model():
    '''
    Load the keras model from disk.

    Returns:
        model: The keras model
    '''
    # load the model
    model: Union[keras.Model, Any] = load_model('model.keras')
    print('Model loaded successfully')

    model.summary()
    return model




if __name__ == '__main__':
    # load the model
    model = load_keras_model()
    print('Model loaded successfully')