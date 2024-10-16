import os, sys
import pickle
import numpy as np
import pandas as pd
from keras.models import load_model
from prepare_dataset import (
    one_hot_encode, 
    pick_most_common_values, 
    apply_common_value_filter,
    basic_cleaning
)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def load_and_preprocess_unlabeled_data(file_path):
    # Load the unlabeled data
    df = pd.read_json(file_path)
    
    # Apply the same preprocessing steps as in prepare_dataset.py
    df = basic_cleaning(df, require_all_features=False)
    
    # Load the pickled file with the features to be used
    with open('features.pkl', 'rb') as f:
        features_used_for_training = pickle.load(f)
    
    feature_columns = df.columns[1:]  # Exclude 'pid'
    for column_name in feature_columns:
        df = one_hot_encode(df, column_name)
    
    ## Ensure the columns match those used in training
    # exclude columns that are not in the trained features
    for column in df.columns:
        if column not in features_used_for_training:
            print(f"Column {column} not in trained features")
            df = df.drop(column, axis=1)

    # add columns that are in the trained features but not in the data
    for column in features_used_for_training:
        if column not in df.columns:
            df[column] = 0
    
    return df

def make_predictions(model, data) -> np.ndarray:
    return model.predict(data)

if __name__ == "__main__":
    # Load the trained model
    model = load_model('model.keras')
    
    # NOTE: For next time:
    # Need to modify the basic_cleaning function to handle situations
    # where the data being cleaned is missing columns

    # Load and preprocess the unlabeled data
    unlabeled_data = load_and_preprocess_unlabeled_data('../source_data/test_record.json')
    
    # Make predictions
    predictions: np.ndarray = make_predictions(model, unlabeled_data)
    
    # Print the predictions
    # Round the predictions to 2 decimal places
    print(np.round(predictions, 2))

    # Process and save the predictions
    # This will depend on how you want to use/interpret the results
    np.save('predictions.npy', predictions)
    
    print("Predictions saved to predictions.npy")