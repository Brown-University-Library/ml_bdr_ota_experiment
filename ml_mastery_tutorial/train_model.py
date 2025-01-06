import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from icecream import ic
import pickle
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import seaborn as sns
from datetime import datetime

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma, alpha):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        
        pos_loss = -y_true * tf.math.log(y_pred)
        neg_loss = -(1 - y_true) * tf.math.log(1 - y_pred)
        
        pos_focal = tf.pow(1 - y_pred, self.gamma) * pos_loss
        neg_focal = tf.pow(y_pred, self.gamma) * neg_loss
        
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

def build_model(n_inputs, n_outputs, gamma, alpha):
    model = keras.Sequential([
        Dense(128, input_dim=n_inputs, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(n_outputs, activation='sigmoid')
    ])
    
    focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=focal_loss,
        metrics=['binary_accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    metrics = []
    for i in range(y_test.shape[1]):
        true_pos = np.sum((y_test[:, i] == 1) & (y_pred_binary[:, i] == 1))
        false_pos = np.sum((y_test[:, i] == 0) & (y_pred_binary[:, i] == 1))
        false_neg = np.sum((y_test[:, i] == 1) & (y_pred_binary[:, i] == 0))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.append({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    avg_precision = np.mean([m['precision'] for m in metrics])
    avg_recall = np.mean([m['recall'] for m in metrics])
    avg_f1 = np.mean([m['f1'] for m in metrics])
    
    return avg_precision, avg_recall, avg_f1

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

def grid_search(X, y, gamma_range, alpha_range):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = []
    total_combinations = len(gamma_range) * len(alpha_range)
    current = 0
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for gamma, alpha in product(gamma_range, alpha_range):
        current += 1
        print(f"\nTesting gamma={gamma}, alpha={alpha} ({current}/{total_combinations})")
        
        model = build_model(X.shape[1], y.shape[1], gamma, alpha)
        
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=30,  # Reduced for grid search
            batch_size=32,
            verbose=0
        )
        
        precision, recall, f1 = evaluate_model(model, X_test, y_test)
        
        results.append({
            'gamma': gamma,
            'alpha': alpha,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        # Save intermediate results
        df = pd.DataFrame(results)
        df.to_csv(f'grid_search_results_{timestamp}.csv', index=False)
        
        # Plot current results
        plot_results(df, timestamp)

    return pd.DataFrame(results)

def plot_results(df, timestamp):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['precision', 'recall', 'f1']
    titles = ['Precision', 'Recall', 'F1 Score']
    
    for ax, metric, title in zip(axes, metrics, titles):
        pivot = df.pivot_table(values=metric, 
                             index='gamma', 
                             columns='alpha', 
                             aggfunc='mean')
        
        sns.heatmap(pivot, ax=ax, cmap='YlOrRd', annot=True, fmt='.3f')
        ax.set_title(title)
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Gamma')
    
    plt.tight_layout()
    plt.savefig(f'grid_search_results_{timestamp}.png')
    plt.close()

if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load dataset
    X, y = open_dataset()

    gamma_range = np.arange(2.0, 5.1, 1.0) # 2 to 6 with step 1
    alpha_range = np.arange(0.1, 0.31, 0.05) # 0.1 to 0.3 with step 0.05
    
    results_df = grid_search(X, y, gamma_range, alpha_range)
    
    # # Load label names if available
    # try:
    #     with open('labels.pkl', 'rb') as f:
    #         label_names = pickle.load(f)
    # except FileNotFoundError:
    #     label_names = None

    # # Analyze distributions
    # analyze_label_distribution(y, label_names)

    
    # # Train model
    # model, scores, label_metrics, history = manage_training(X, y, label_names)
    # model.save('model.keras')
    
    # # Save training history
    # np.save('training_history.npy', history.history)
    # print('\nModel and training history saved')