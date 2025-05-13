import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss

CSV_PATH = 'dog_centroid_classification.csv'
NOVEL_LABEL = 'novel'  # Use this for anything below the low threshold

# Load data
print(f'Loading {CSV_PATH}...')
df = pd.read_csv(CSV_PATH)

# Map true_subclass >= 100 to 'novel' for evaluation
true_subclass_eval = df['true_subclass'].apply(lambda x: NOVEL_LABEL if int(x) >= 100 else str(x))
df['true_subclass_eval'] = true_subclass_eval

# Set low threshold and iterate only over high thresholds
highs = np.arange(0.90, 0.99, 0.01)
low = 0.80

for high in highs:
    preds = []
    for idx, row in df.iterrows():
        if row['pred_cosine'] >= high:
            preds.append(str(row['pred_subclass']))
        elif row['pred_cosine'] < low:
            preds.append(NOVEL_LABEL)
        else:
            preds.append('unsure')
    df['threshold_pred'] = preds
    # Metrics for known subclasses and novel (ignore unsure)
    eval_mask = (df['threshold_pred'] != 'unsure')
    y_true = df.loc[eval_mask, 'true_subclass_eval']
    y_pred = df.loc[eval_mask, 'threshold_pred']
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=3, zero_division=0)
    # cm = confusion_matrix(y_true, y_pred)
    unsure_count = (df['threshold_pred'] == 'unsure').sum()
    novel_count = (df['threshold_pred'] == NOVEL_LABEL).sum()
    print(f'High={high:.2f} Low={low:.2f} | unsure={unsure_count} | novel={novel_count} | acc={acc:.3f}')
    # print(report)
    # print('Confusion Matrix:')
    # print(cm)
    # print('-'*60) 