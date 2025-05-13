import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from clip_client import Client
from sklearn.metrics.pairwise import cosine_similarity
import base64
import h5py
import pandas as pd

# Add phase1 to sys.path for EmbeddingStorage
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'phase1')))
from embedding_storage import EmbeddingStorage

# Paths
CENTROIDS_PATH = os.path.join(os.path.dirname(__file__), '../phase1/data/class_centroids.npy')
CLASSES = ['Bird', 'Dog', 'Reptile']
ALL_CLASSES = ['Bird', 'Dog', 'Reptile', 'novel', 'unsure']
HIGH_THRESHOLD = 0.90
LOW_THRESHOLD = 0.72

# Embedding extraction
BATCH_SIZE = 20

LABEL_MAP = {'Bird': 0, 'Dog': 1, 'Reptile': 2, 'novel': 3, 'unsure': 'unsure'}

def create_clip_client():
    TCP_URL = '0.tcp.ngrok.io:13104'
    return Client(f'grpc://{TCP_URL}')

def encode_image_to_base64(img_path: str) -> str:
    with open(img_path, 'rb') as image_file:
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        return f'data:image/jpeg;base64,{base64_image}'

def process_batch_with_retry(client, image_paths, max_retries=3):
    for attempt in range(max_retries):
        try:
            batch_uris = [encode_image_to_base64(path) for path in image_paths]
            return client.encode(batch_uris)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"\nError processing batch (attempt {attempt + 1}/{max_retries}): {str(e)}")
            print("Reconnecting to CLIP server...")
            client = create_clip_client()
    return None

def extract_embeddings(image_folder, dev_mode=False):
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jpeg'))]
    image_files.sort()
    if dev_mode:
        print("[DEV MODE] Only processing first 10 images.")
        image_files = image_files[:10]
    image_paths = [os.path.join(image_folder, f) for f in image_files]
    embeddings = []
    valid_image_paths = []
    client = create_clip_client()
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc='Extracting embeddings'):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        try:
            batch_embeddings = process_batch_with_retry(client, batch_paths)
            embeddings.extend(batch_embeddings)
            valid_image_paths.extend(batch_paths)
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {str(e)}")
            print("Skipping batch...")
    return np.array(embeddings), valid_image_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='Folder containing images to classify')
    parser.add_argument('--output_dir', required=True, help='Directory to save outputs')
    parser.add_argument('--dev', action='store_true', help='Run in dev mode (only process 10 images)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, 'superclass_predictions.csv')

    # Step 1: Extract embeddings
    embeddings, image_paths = extract_embeddings(args.image_folder, dev_mode=args.dev)
    print(f"Extracted embeddings for {len(image_paths)} images.")

    # Step 2: Load centroids
    centroids = np.load(CENTROIDS_PATH, allow_pickle=True).item()
    centroids = {k: v for k, v in centroids.items() if k in CLASSES}

    # Step 3: Predict labels
    predictions = []
    class_to_indices = {cls: [] for cls in ALL_CLASSES}
    for idx, (emb, img_path) in enumerate(zip(embeddings, image_paths)):
        sims = {cls: cosine_similarity(emb.reshape(1, -1), centroid.reshape(1, -1))[0, 0] for cls, centroid in centroids.items()}
        best_class = max(sims, key=sims.get)
        best_score = sims[best_class]
        if best_score >= HIGH_THRESHOLD:
            pred = best_class
        elif best_score < LOW_THRESHOLD:
            pred = 'novel'
        else:
            pred = 'unsure'
        pred_idx = LABEL_MAP[pred]
        predictions.append({'image_id': os.path.basename(img_path), 'phase1_pred_superclass': pred_idx})
        class_to_indices[pred].append(idx)

    # Step 4: Save predictions to CSV
    df = pd.DataFrame(predictions)
    df.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")

    # Step 5: Save all embeddings and image ids in a single HDF5 file
    all_h5 = os.path.join(args.output_dir, 'all_embeddings.h5')
    with h5py.File(all_h5, 'w') as f:
        f.create_dataset('embeddings', data=embeddings)
        image_paths_ascii = [str(p).encode('utf-8') for p in image_paths]
        f.create_dataset('image_paths', data=image_paths_ascii)
    print(f"Saved all embeddings to {all_h5}")

if __name__ == "__main__":
    main() 