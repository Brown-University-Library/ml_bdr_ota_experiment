import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from icecream import ic
import pickle

class FocalLoss(tf.keras.losses.Loss):
    """
    Implements focal loss for multi-label classification
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # Convert types to float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Clip predictions to prevent log(0)
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        
        # Calculate cross entropy for both positive and negative cases
        pos_loss = -y_true * tf.math.log(y_pred)
        neg_loss = -(1 - y_true) * tf.math.log(1 - y_pred)
        
        # Apply focal modulation
        pos_focal = tf.pow(1 - y_pred, self.gamma) * pos_loss
        neg_focal = tf.pow(y_pred, self.gamma) * neg_loss
        
        # Apply alpha weighting
        loss = self.alpha * pos_focal + (1 - self.alpha) * neg_focal
        
        return tf.reduce_mean(loss)

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
    Creates a model using focal loss
    """
    model = keras.Sequential([
        # Input layer
        Dense(128, input_dim=n_inputs, activation='relu'),
        Dropout(0.3),
        
        # Hidden layer
        Dense(128, activation='relu'),
        Dropout(0.3),
        
        # Output layer - one neuron per label
        Dense(n_outputs, activation='sigmoid')
    ])
    
    # Using focal loss
    '''
    In focal loss, alpha is a simple class weighting factor 
    (like 0.25 vs 0.75 for positive/negative examples), while gamma 
    dynamically reduces the loss contribution from "easy" examples by 
    making correctly classified examples contribute less and less as the 
    model becomes more confident about them.
    '''
    # focal_loss = FocalLoss(gamma=2.0, alpha=0.25) # Initial suggested values - Recall 100% Precision 7%
    # focal_loss = FocalLoss(gamma=4.0, alpha=0.1) # Recall: 18.51% Precision: 95.91%
    # focal_loss = FocalLoss(gamma=3.0, alpha=0.1) # Recall: 22.97% Precision: 94.00%
    focal_loss = FocalLoss(gamma=3.0, alpha=0.15) # Recall: 27.06% Precision: 92.31%

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=focal_loss,
        metrics=[
            'binary_accuracy',
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision()
        ]
    )
    return model

def analyze_label_distribution(y, label_names=None):
    """
    Analyze the distribution and raw counts of each label in the dataset.
    
    Args:
        y: numpy array of shape (n_samples, n_labels) containing the labels
        label_names: optional list of label names corresponding to each column index
    """
    n_samples, n_labels = y.shape
    
    print("\nLabel Distribution Analysis:")
    print("-" * 60)
    print(f"Total number of samples: {n_samples}")
    print("\nPer-label statistics:")
    print("-" * 60)
    
    for i in range(n_labels):
        positive_count = np.sum(y[:, i] == 1)
        negative_count = np.sum(y[:, i] == 0)
        positive_ratio = positive_count / n_samples * 100
        
        label_name = f"Label {i}" if label_names is None else label_names[i]
        
        print(f"\n{label_name}:")
        print(f"  Positive samples: {positive_count:,}")
        print(f"  Negative samples: {negative_count:,}")
        print(f"  Positive ratio: {positive_ratio:.2f}%")
    
    # Analysis of samples with multiple labels
    label_counts_per_sample = np.sum(y, axis=1)
    avg_labels = np.mean(label_counts_per_sample)
    max_labels = np.max(label_counts_per_sample)
    min_labels = np.min(label_counts_per_sample)
    
    print("\nMulti-label statistics:")
    print("-" * 60)
    print(f"Average labels per sample: {avg_labels:.2f}")
    print(f"Maximum labels per sample: {max_labels:.0f}")
    print(f"Minimum labels per sample: {min_labels:.0f}")
    
    # Distribution of number of labels per sample
    unique_counts = np.unique(label_counts_per_sample, return_counts=True)
    print("\nDistribution of labels per sample:")
    for count, freq in zip(unique_counts[0], unique_counts[1]):
        percentage = (freq / n_samples) * 100
        print(f"  {count} labels: {freq:,} samples ({percentage:.2f}%)")


def evaluate_multilabel(model, X_test, y_test, label_names=None):
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
    print("-" * 60)
    
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
        
        label_name = f"Label {i}" if label_names is None else label_names[i]
        print(f"\n{label_name}:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1: {f1:.3f}")
        
        # Add support (number of positive examples)
        support = np.sum(y_test[:, i] == 1)
        print(f"  Support: {support}")
    
    # Calculate overall metrics
    scores = model.evaluate(X_test, y_test, verbose=0)
    
    print("\nOverall Model Evaluation:")
    print("-" * 40)
    print(f'Loss: {scores[0]:.4f}')
    print(f'Binary Accuracy: {scores[1]*100:.2f}%')
    print(f'Recall: {scores[2]*100:.2f}%')
    print(f'Precision: {scores[3]*100:.2f}%')
    
    return scores, label_metrics

def manage_training(X, y, label_names=None):
    """
    Manage the training process
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
    scores, label_metrics = evaluate_multilabel(model, X_test, y_test, label_names)
    
    return model, scores, label_metrics, history

if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load dataset
    X, y = open_dataset()
    
    # Load label names if available
    try:
        with open('labels.pkl', 'rb') as f:
            label_names = pickle.load(f)
    except FileNotFoundError:
        label_names = None

    # Analyze distributions
    analyze_label_distribution(y, label_names)

    
    # Train model
    model, scores, label_metrics, history = manage_training(X, y, label_names)
    model.save('model.keras')
    
    # Save training history
    np.save('training_history.npy', history.history)
    print('\nModel and training history saved')