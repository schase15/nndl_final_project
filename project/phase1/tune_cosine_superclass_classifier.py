"""
This script uses cosine similarity to classify embeddings as Bird, Dog, or Reptile using centroids.
It applies two thresholds: above high = class, below low = novel, between = unsure.
Uses the training embeddings and labels for evaluation.
Now maps true labels to class names, treats anything >2 as novel, and computes precision/recall/f1/confusion matrix.
Sweeps over a grid of thresholds to help tune for fewer 'unsure' predictions while maintaining high recall/precision.
Also computes overall accuracy for known classes and categorical cross-entropy.
Now uses the new combined_train_embeddings.h5 file for data.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
from embedding_storage import EmbeddingStorage

# Paths
CENTROIDS_PATH = 'data/class_centroids.npy'
TRAIN_H5_PATH = 'data/combined_train_embeddings.h5'

# Classes of interest
CLASSES = ['Bird', 'Dog', 'Reptile']
ALL_CLASSES = ['Bird', 'Dog', 'Reptile', 'novel']

# Label mapping
LABEL_TO_NAME = {0: 'Bird', 1: 'Dog', 2: 'Reptile'}

# For log_loss, need to map class names to indices
CLASS_NAME_TO_IDX = {name: i for i, name in enumerate(ALL_CLASSES)}

def map_true_label(label):
    if label in LABEL_TO_NAME:
        return LABEL_TO_NAME[label]
    else:
        return 'novel'

# Load centroids and training data
centroids = np.load(CENTROIDS_PATH, allow_pickle=True).item()
train_embeddings, train_labels, train_image_paths = EmbeddingStorage.load(TRAIN_H5_PATH)

# Only keep centroids for Bird, Dog, Reptile
centroids = {k: v for k, v in centroids.items() if k in CLASSES}

# Threshold grid to sweep
high_thresholds = np.arange(0.80, 0.91, 0.02)  # e.g. 0.80, 0.82, ..., 0.90
low_thresholds = np.arange(0.60, 0.76, 0.03)   # e.g. 0.60, 0.63, ..., 0.75

best_config = None
best_f1 = 0

for HIGH_THRESHOLD in high_thresholds:
    for LOW_THRESHOLD in low_thresholds:
        if LOW_THRESHOLD >= HIGH_THRESHOLD:
            continue
        results = []
        y_true_known = []
        y_pred_known = []
        y_true_all = []
        y_pred_all = []
        y_prob_all = []
        for emb, true_label in zip(train_embeddings, train_labels):
            sims = {cls: cosine_similarity(emb.reshape(1, -1), centroid.reshape(1, -1))[0, 0] for cls, centroid in centroids.items()}
            best_class = max(sims, key=sims.get)
            best_score = sims[best_class]
            if best_score >= HIGH_THRESHOLD:
                pred = best_class
            elif best_score < LOW_THRESHOLD:
                pred = 'novel'
            else:
                pred = 'unsure'
            mapped_true = map_true_label(true_label)
            results.append({'true_label': mapped_true, 'predicted': pred, 'best_class': best_class, 'best_score': best_score, **sims})
            # For accuracy and log_loss
            if pred != 'unsure':
                y_true_all.append(mapped_true)
                y_pred_all.append(pred)
                # For log_loss, create a probability vector for all classes
                prob_vec = np.zeros(len(ALL_CLASSES))
                # Softmax over similarities for known classes
                sim_values = np.array([sims.get(cls, 0) for cls in CLASSES])
                exp_sim = np.exp(sim_values)
                softmax_sim = exp_sim / np.sum(exp_sim)
                for i, cls in enumerate(CLASSES):
                    prob_vec[CLASS_NAME_TO_IDX[cls]] = softmax_sim[i]
                # If predicted novel, set novel prob to 1
                if pred == 'novel':
                    prob_vec[CLASS_NAME_TO_IDX['novel']] = 1.0
                y_prob_all.append(prob_vec)
                # For known class accuracy
                if mapped_true in CLASSES:
                    y_true_known.append(mapped_true)
                    y_pred_known.append(pred)
        results_df = pd.DataFrame(results)
        filtered = results_df[results_df['predicted'] != 'unsure']
        y_true = filtered['true_label']
        y_pred = filtered['predicted']
        report = classification_report(y_true, y_pred, labels=ALL_CLASSES, zero_division=0, output_dict=True)
        macro_f1 = report['macro avg']['f1-score']
        unsure_count = (results_df['predicted'] == 'unsure').sum()
        # Accuracy for known classes
        if y_true_known:
            known_acc = accuracy_score(y_true_known, y_pred_known)
        else:
            known_acc = 0.0
        # Categorical cross-entropy (log_loss)
        if y_true_all and y_prob_all:
            y_true_idx = [CLASS_NAME_TO_IDX[y] for y in y_true_all]
            logloss = log_loss(y_true_idx, np.array(y_prob_all), labels=list(range(len(ALL_CLASSES))))
        else:
            logloss = float('nan')
        print(f"HIGH={HIGH_THRESHOLD:.2f} LOW={LOW_THRESHOLD:.2f} | unsure={unsure_count:4d} | macro F1={macro_f1:.3f} | known acc={known_acc:.3f} | logloss={logloss:.3f}")
        for cls in ALL_CLASSES:
            print(f"  {cls:8s}  prec={report[cls]['precision']:.2f}  rec={report[cls]['recall']:.2f}  f1={report[cls]['f1-score']:.2f}")
        print()
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_config = (HIGH_THRESHOLD, LOW_THRESHOLD, unsure_count, known_acc, logloss)

print(f"Best config: HIGH={best_config[0]:.2f} LOW={best_config[1]:.2f} | unsure={best_config[2]} | macro F1={best_f1:.3f} | known acc={best_config[3]:.3f} | logloss={best_config[4]:.3f}")

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('results/cosine_superclass_classification_train_results.csv', index=False)

# Filter out 'unsure' for metrics
filtered = results_df[results_df['predicted'] != 'unsure']

# Compute metrics
y_true = filtered['true_label']
y_pred = filtered['predicted']
print('\nClassification Report (excluding unsure):')
print(classification_report(y_true, y_pred, labels=ALL_CLASSES, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=ALL_CLASSES)
cm_df = pd.DataFrame(cm, index=ALL_CLASSES, columns=ALL_CLASSES)
print('\nConfusion Matrix (excluding unsure):')
print(cm_df)

# Print summary
print('\nPrediction counts:')
print(results_df['predicted'].value_counts())
print('\nSample results:')
print(results_df.head()) 