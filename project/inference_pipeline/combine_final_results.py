import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True, help='Directory containing superclass and subclass prediction CSVs')
    args = parser.parse_args()

    results_dir = args.results_dir
    superclass_csv = os.path.join(results_dir, 'superclass_predictions.csv')
    if not os.path.exists(superclass_csv):
        raise FileNotFoundError(f"{superclass_csv} not found")

    # Load superclass predictions
    super_df = pd.read_csv(superclass_csv)
    # Use the final_superclass_pred as the superclass index (convert to int)
    super_df['superclass_index'] = super_df['final_superclass_pred'].astype(float).astype(int)

    # Prepare a dict to hold subclass predictions for each class
    subclass_pred_files = [f for f in os.listdir(results_dir) if f.endswith('_subclass_predictions.csv')]
    subclass_dfs = []
    for fname in subclass_pred_files:
        class_name = fname.split('_')[0]
        df = pd.read_csv(os.path.join(results_dir, fname))

        # Only keep image_id and final_subclass_pred
        df = df[['image_id', 'final_subclass_pred']].rename(columns={'final_subclass_pred': 'subclass_index'})
        subclass_dfs.append(df)
    # Concatenate all subclass predictions
    subclass_df = pd.concat(subclass_dfs, ignore_index=True)

    # Merge on image_id
    merged = pd.merge(super_df, subclass_df, on='image_id', how='left')
    # Select only the required columns
    final_df = merged[['image_id', 'superclass_index', 'subclass_index']]
    # Save to final_results.csv
    out_csv = os.path.join(results_dir, 'final_results.csv')
    
    # Rename image_id to image
    final_df = final_df.rename(columns={'image_id': 'image'})
    
    # Set subclass_index to 87 if superclass_index is 3 (novel)
    final_df.loc[final_df['superclass_index'] == 3, 'subclass_index'] = 87

    # Ensure subclass_index is int
    final_df['subclass_index'] = final_df['subclass_index'].astype(int)

    final_df.to_csv(out_csv, index=False)
    print(f"Saved final results to {out_csv}")

if __name__ == '__main__':
    main() 