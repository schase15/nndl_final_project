import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
import numpy as np

SUPERCLASSES = ['dog', 'bird', 'reptile']

for superclass in SUPERCLASSES:
    csv_path = f'{superclass}_centroid_classification.csv'
    metrics_path = f'{superclass}_centroid_metrics.txt'
    if not os.path.exists(csv_path):
        print(f'File not found: {csv_path}')
        continue
    df = pd.read_csv(csv_path)
    y_true = df['true_subclass']
    y_pred = df['pred_subclass']
    # Classification report
    report = classification_report(y_true, y_pred, digits=4)
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Subclass accuracy
    acc = accuracy_score(y_true, y_pred)
    # Subclass cross-entropy (log loss)
    # Get all centroid columns
    centroid_cols = [col for col in df.columns if col.startswith('cosine_')]
    # Convert cosine similarities to probabilities (softmax)
    logits = df[centroid_cols].values.astype(float)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    # Map true labels to indices in centroid_cols
    label_to_index = {int(col.split('_')[1]): i for i, col in enumerate(centroid_cols)}
    y_true_indices = df['true_subclass'].map(label_to_index).values
    # Compute log loss
    ce = log_loss(y_true_indices, probs)
    # Save metrics
    with open(metrics_path, 'w') as f:
        f.write(f'Classification Report for {superclass}\n')
        f.write(report + '\n')
        f.write('Confusion Matrix:\n')
        f.write(pd.DataFrame(cm).to_string())
        f.write(f'\n\nSubclass Accuracy: {acc:.4f}\n')
        f.write(f'Subclass Cross-Entropy (Log Loss): {ce:.4f}\n')
    print(f'Saved metrics to {metrics_path}') 