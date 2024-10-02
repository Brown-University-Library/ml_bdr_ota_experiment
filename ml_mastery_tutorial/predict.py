import numpy as np
import pandas as pd
from keras.models import load_model
from prepare_dataset import (
    one_hot_encode, 
    pick_most_common_values, 
    apply_common_value_filter,
    basic_cleaning
)

def load_and_preprocess_unlabeled_data(file_path):
    # Load the unlabeled data
    df = pd.read_json(file_path)
    
    # Apply the same preprocessing steps as in prepare_dataset.py
    df = basic_cleaning(df)
    
    # Apply the same feature engineering steps
    top_keyword_values = pick_most_common_values(df, 'keyword', 1000)
    df = df.apply(lambda x: apply_common_value_filter(x, 'keyword', top_keyword_values), axis=1)
    
    feature_columns = df.columns[1:]  # Exclude 'pid'
    for column_name in feature_columns:
        df = one_hot_encode(df, column_name)
    
    # Ensure the columns match those used in training
    # You might need to add missing columns or remove extra ones
    
    return df

def make_predictions(model, data):
    return model.predict(data)

if __name__ == "__main__":
    # Load the trained model
    model = load_model('model.keras')
    
    # Load and preprocess the unlabeled data
    unlabeled_data = load_and_preprocess_unlabeled_data('path_to_unlabeled_data.json')
    
    # Make predictions
    predictions = make_predictions(model, unlabeled_data)
    
    # Process and save the predictions
    # This will depend on how you want to use/interpret the results
    np.save('predictions.npy', predictions)
    
    print("Predictions saved to predictions.npy")