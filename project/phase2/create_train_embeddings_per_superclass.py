import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import base64
from typing import List, Tuple
from clip_client import Client
from pathlib import Path
import time

# Add phase1 to sys.path for EmbeddingStorage
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'phase1')))
from embedding_storage import EmbeddingStorage

SUPERCLASSES = {
    'dog': 1,
    'bird': 0,
    'reptile': 2
}

TRAIN_IMAGES_DIR = os.path.join('..', 'data', 'Released_Data_NNDL_2025', 'train_images')
TRAIN_CSV = os.path.join('..', 'data', 'Released_Data_NNDL_2025', 'train_data.csv')

BATCH_SIZE = 20

# CLIP server connection
def create_clip_client():
    TCP_URL = '0.tcp.ngrok.io:19747'
    return Client(f'grpc://{TCP_URL}')

def encode_image_to_base64(img_path: str) -> str:
    with open(img_path, 'rb') as image_file:
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        return f'data:image/jpeg;base64,{base64_image}'

def process_batch_with_retry(client: Client, image_paths: List[str], max_retries: int = 3) -> np.ndarray:
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
            time.sleep(2 ** attempt)
    return None

def process_images_batch(image_paths: List[str], batch_size: int = 20) -> List[Tuple[np.ndarray, str]]:
    results = []
    client = create_clip_client()
    progress_bar = tqdm(range(0, len(image_paths), batch_size))
    for i in progress_bar:
        batch_paths = image_paths[i:i+batch_size]
        try:
            batch_embeddings = process_batch_with_retry(client, batch_paths)
            results.extend(zip(batch_embeddings, batch_paths))
            progress_bar.set_description(f"Processed {i + len(batch_paths)}/{len(image_paths)} images")
        except Exception as e:
            print(f"\nError processing batch starting at index {i}: {str(e)}")
            print("Saving progress and exiting...")
            return results
    return results

def save_progress(storage: EmbeddingStorage, output_file: str):
    print("\nSaving current progress...")
    storage.save()
    embeddings, labels, paths = EmbeddingStorage.load(output_file)
    print("\nCurrent progress statistics:")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of images: {len(paths)}")
    print(f"Images per class: {pd.Series(labels).value_counts().sort_index()}")
    print("Progress saved successfully!")

# Main logic
train_df = pd.read_csv(TRAIN_CSV)

for superclass_name, superclass_idx in SUPERCLASSES.items():
    print(f'Processing {superclass_name}...')
    superclass_df = train_df[train_df['superclass_index'] == superclass_idx]
    subclass_counts = superclass_df['subclass_index'].value_counts()
    storage = EmbeddingStorage(f'train_{superclass_name}.h5')
    for subclass_idx in subclass_counts.index:
        subclass_df = superclass_df[superclass_df['subclass_index'] == subclass_idx]
        # Take up to 300 images per subclass
        subclass_sample = subclass_df.sample(n=min(300, len(subclass_df)), random_state=42)
        image_paths = [os.path.join(TRAIN_IMAGES_DIR, row["image"]) for _, row in subclass_sample.iterrows()]
        # Process in batches using CLIP server
        results = process_images_batch(image_paths, batch_size=BATCH_SIZE)
        for embedding, img_path in results:
            storage.add_sample(embedding, int(subclass_idx), os.path.basename(img_path))
    storage.save()
    print(f'Saved train_{superclass_name}.h5') 