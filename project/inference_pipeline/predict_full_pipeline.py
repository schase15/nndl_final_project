import os
import argparse
import subprocess
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, log_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True, help='Directory of images to classify')
    parser.add_argument('--output_dir', required=True, help='Directory to save all outputs')
    parser.add_argument('--dev', action='store_true', help='Run in dev mode (faster, fewer images)')
    parser.add_argument('--true_labels', required=False, help='CSV file with true labels for evaluation (optional)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Run predict_superclass.py
    superclass_cmd = [
        'python', os.path.join(os.path.dirname(__file__), 'predict_superclass.py'),
        '--image_dir', args.image_dir,
        '--output_dir', args.output_dir
    ]
    if args.dev:
        superclass_cmd.append('--dev')
    print('Running predict_superclass.py...')
    # subprocess.run(superclass_cmd, check=True)

    # 2. Run predict_subclass.py
    superclass_csv = os.path.join(args.output_dir, 'superclass_predictions.csv')
    subclass_cmd = [
        'python', os.path.join(os.path.dirname(__file__), 'predict_subclass.py'),
        '--csv', superclass_csv,
        '--image_dir', args.image_dir,
        '--output_dir', args.output_dir
    ]
    if args.dev:
        subclass_cmd.append('--dev')
    print('Running predict_subclass.py...')
    subprocess.run(subclass_cmd, check=True)

    # 3. Run combine_final_results.py
    combine_cmd = [
        'python', os.path.join(os.path.dirname(__file__), 'combine_final_results.py'),
        '--results_dir', args.output_dir
    ]
    print('Combining results...')
    subprocess.run(combine_cmd, check=True)

    final_csv = os.path.join(args.output_dir, 'final_results.csv')
    print(f'Pipeline complete! Final results: {final_csv}')

    # --- Metrics Calculation ---
    if args.true_labels:
        import pandas as pd
        os.makedirs(os.path.join(args.output_dir, 'metrics'), exist_ok=True)
        metrics_path = os.path.join(args.output_dir, 'metrics', 'metrics.txt')
        pred_df = pd.read_csv(final_csv)
        true_df = pd.read_csv(args.true_labels)
        # Expect columns: image_id, superclass_index, subclass_index
        merged = pd.merge(pred_df, true_df, on='image', suffixes=('_pred', '_true'))

        metrics = []

        # convert subclass_index_pred to int
        merged['subclass_index_pred'] = merged['subclass_index_pred'].astype(int)

        for label_type in ['superclass', 'subclass']:
            print(f'Metrics for {label_type}...')
            y_true = merged[f'{label_type}_index_true']
            y_pred = merged[f'{label_type}_index_pred']
            # Precision, recall, f1, accuracy
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            print(f'Precision: {precision}')
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            print(f'Recall: {recall}')
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            print(f'F1: {f1}')
            accuracy = accuracy_score(y_true, y_pred)
            print(f'Accuracy: {accuracy}')
            # Categorical cross-entropy (log loss) using one-hot encoding
            y_true_oh = np.eye(np.max(y_true)+1)[y_true]
            y_pred_oh = np.eye(np.max(y_true)+1)[y_pred]
            try:
                logloss = log_loss(y_true_oh, y_pred_oh)
            except Exception as e:
                logloss = float('nan')
            print(f'Categorical Cross-Entropy: {logloss}')

            # Accuracy breakdowns
            if label_type == 'superclass':
                novel_val = 3
            else:
                novel_val = 87
            seen_mask = y_true != novel_val
            novel_mask = y_true == novel_val
            accuracy_seen = accuracy_score(y_true[seen_mask], y_pred[seen_mask]) if seen_mask.any() else float('nan')
            accuracy_novel = accuracy_score(y_true[novel_mask], y_pred[novel_mask]) if novel_mask.any() else float('nan')
            print(f'Accuracy (seen): {accuracy_seen}')
            print(f'Accuracy (novel): {accuracy_novel}')

            metrics.append((label_type, precision, recall, f1, accuracy, logloss, accuracy_seen, accuracy_novel))

        with open(metrics_path, 'w') as f:
            f.write('METRICS REPORT\n')
            f.write('====================\n')
            for label_type, precision, recall, f1, accuracy, logloss, accuracy_seen, accuracy_novel in metrics:
                f.write(f'{label_type.upper()} METRICS:\n')
                f.write(f'  Precision:                {precision:.4f}\n')
                f.write(f'  Recall:                   {recall:.4f}\n')
                f.write(f'  F1 Score:                 {f1:.4f}\n')
                f.write(f'  Accuracy (overall):       {accuracy:.4f}\n')
                f.write(f'  Accuracy (seen):          {accuracy_seen:.4f}\n')
                f.write(f'  Accuracy (novel):         {accuracy_novel:.4f}\n')
                f.write(f'  Categorical Cross-Entropy: {logloss:.4f}\n')
                f.write('--------------------\n')
            f.write('\n')
        print(f'Metrics written to {metrics_path}')

if __name__ == '__main__':
    main() 