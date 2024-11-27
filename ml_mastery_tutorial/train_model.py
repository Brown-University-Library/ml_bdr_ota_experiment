import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from icecream import ic

def open_dataset():
    '''
    Open the dataset, stored as a npz file.
    '''
    data = np.load('dataset.npz')
    X, y = data['X'], data['y']
    ic(X.shape, y.shape)
    return X, y

def build_model(n_inputs, n_outputs):
    """
    Creates a model specifically designed for multi-label classification
    """
    model = keras.Sequential([
        # Input layer
        Dense(128, input_dim=n_inputs, activation='relu'),
        Dropout(0.3),
        
        # Hidden layer - keeping network wider for multi-label
        Dense(128, activation='relu'),
        Dropout(0.3),
        
        # Output layer - one neuron per label
        Dense(n_outputs, activation='sigmoid')
    ])
    
    # Using binary crossentropy since each output is a binary classification
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'binary_accuracy',
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision()
        ]
    )
    return model

def evaluate_multilabel(model, X_test, y_test):
    """
    Evaluate the model with multi-label specific metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate per-label metrics
    n_labels = y_test.shape[1]
    label_metrics = []
    
    print("\nPer-label metrics:")
    print("-" * 40)
    
    for i in range(n_labels):
        true_pos = np.sum((y_test[:, i] == 1) & (y_pred_binary[:, i] == 1))
        false_pos = np.sum((y_test[:, i] == 0) & (y_pred_binary[:, i] == 1))
        false_neg = np.sum((y_test[:, i] == 1) & (y_pred_binary[:, i] == 0))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        label_metrics.append({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        print(f"Label {i}:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1: {f1:.3f}")
    
    # Calculate overall metrics
    scores = model.evaluate(X_test, y_test, verbose=0)
    
    print("\nOverall Model Evaluation:")
    print("-" * 40)
    print(f'Loss: {scores[0]:.4f}')
    print(f'Binary Accuracy: {scores[1]*100:.2f}%')
    print(f'Recall: {scores[2]*100:.2f}%')
    print(f'Precision: {scores[3]*100:.2f}%')
    
    return scores, label_metrics

def manage_training(X, y):
    """
    Manage the training process for multi-label classification
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    print(f'Input shape: {X.shape}, Output shape: {y.shape}')
    
    # Build model
    model = build_model(n_inputs, n_outputs)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    print('Training model...')
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate with multi-label metrics
    print('\nEvaluating model...')
    scores, label_metrics = evaluate_multilabel(model, X_test, y_test)
    
    return model, scores, label_metrics, history

if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    
    X, y = open_dataset()
    model, scores, label_metrics, history = manage_training(X, y)
    model.save('model.keras')
    
    # Save training history
    np.save('training_history.npy', history.history)
    print('\nModel and training history saved')

    print('--'*20)
    print('We determined that the model is not predicting well')
    print('There is a large imbalance in the data, where some broad themes are')
    print('overrepresented and others are underrepresented')
    print('We need to look into techniques to handle imbalanced data')
    print('--'*20)