import os
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import argparse

SUPERCLASS_MAP = {0: 'bird', 1: 'dog', 2: 'reptile'}
THRESH_HIGH = 0.95
THRESH_LOW = 0.8
NOVEL_SUBCLASS_IDX = 87

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--superclass', type=int, required=True, help='Index of the superclass to process')
    parser.add_argument('--csv', required=True, help='CSV from predict_superclass (superclass_predictions.csv)')
    parser.add_argument('--image_dir', required=True, help='Directory of images to classify')
    parser.add_argument('--output_dir', required=True, help='Directory to save all outputs')
    parser.add_argument('--dev', action='store_true', help='Run in dev mode (faster, fewer images)')
    args = parser.parse_args()

    if args.superclass not in SUPERCLASS_MAP:
        raise ValueError(f"Superclass index {args.superclass} not in SUPERCLASS_MAP: {list(SUPERCLASS_MAP.keys())}")
    idx = args.superclass
    class_name = SUPERCLASS_MAP[idx]

    # Prepare output directory
    subclass_pred_dir = os.path.join(args.output_dir, 'subclass_predictions')
    os.makedirs(subclass_pred_dir, exist_ok=True)
    out_csv = os.path.join(subclass_pred_dir, f'{class_name}_subclass_predictions.csv')

    # Load embeddings and image ids for this superclass
    h5_path = os.path.join('results', 'superclass_subsets', f'class_{idx}.h5')
    with h5py.File(h5_path, 'r') as f:
        embeddings = f['embeddings'][:]
        image_paths = [p.decode('utf-8') if isinstance(p, bytes) else str(p) for p in f['image_paths'][:]]

    # Load centroids
    centroid_dir = os.path.join('../phase2', class_name)
    centroids = {}
    for fname in os.listdir(centroid_dir):
        if fname.endswith('_centroid.npy'):
            subclass_idx = int(fname.split('_')[0])
            centroids[subclass_idx] = np.load(os.path.join(centroid_dir, fname))
    centroid_keys = sorted(centroids.keys())
    centroid_arr = np.stack([centroids[k] for k in centroid_keys])
    # Classify
    preds = []
    for emb, img_id in zip(embeddings, image_paths):
        emb = emb.reshape(1, -1)
        cos_sims = cosine_similarity(emb, centroid_arr)[0]
        pred_idx = np.argmax(cos_sims)
        max_sim = cos_sims[pred_idx]
        if max_sim >= THRESH_HIGH:
            pred_subclass = centroid_keys[pred_idx]
        elif max_sim < THRESH_LOW:
            pred_subclass = NOVEL_SUBCLASS_IDX
        else:
            pred_subclass = 'unsure'
        preds.append({'image_id': os.path.basename(img_id), 'phase1_pred_subclass': pred_subclass})
    df = pd.DataFrame(preds)
    df.to_csv(out_csv, index=False)
    print(f'Saved {len(df)} predictions to {out_csv}') 