"""
This script processes images from the imagenet-organized dataset and creates CLIP embeddings.
It uses the mapping created by create_imagenet_mapping.py and follows the same
robust processing approach used for the NNDL dataset.
"""

import os
import base64
import numpy as np
import pandas as pd
from pathlib import Path
from clip_client import Client
from tqdm import tqdm
from embedding_storage import EmbeddingStorage
import time
from typing import List, Tuple
import sys

def create_clip_client():
    """Create a new CLIP client connection."""
    TCP_URL = '2.tcp.ngrok.io:11737'  # Update this with your current ngrok URL
    return Client(f'grpc://{TCP_URL}')

def encode_image_to_base64(img_path: str) -> str:
    """Encode a single image to base64."""
    with open(img_path, 'rb') as image_file:
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        return f'data:image/jpeg;base64,{base64_image}'

def process_batch_with_retry(client: Client, image_paths: List[str], max_retries: int = 3) -> np.ndarray:
    """Process a single batch with retry logic."""
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
            time.sleep(2 ** attempt)  # Exponential backoff
    return None

def process_images_batch(image_paths: List[str], batch_size: int = 20) -> List[Tuple[np.ndarray, str]]:
    """Process a list of image paths in batches and return their embeddings with paths."""
    results = []
    client = create_clip_client()
    
    progress_bar = tqdm(range(0, len(image_paths), batch_size))
    for i in progress_bar:
        batch_paths = image_paths[i:i+batch_size]
        try:
            batch_embeddings = process_batch_with_retry(client, batch_paths)
            results.extend(zip(batch_embeddings, batch_paths))
            
            # Update progress message
            progress_bar.set_description(f"Processed {i + len(batch_paths)}/{len(image_paths)} images")
            
        except Exception as e:
            print(f"\nError processing batch starting at index {i}: {str(e)}")
            print("Saving progress and exiting...")
            return results
    
    return results

def save_progress(storage: EmbeddingStorage, output_file: str):
    """Save current progress and verify the save."""
    print("\nSaving current progress...")
    storage.save()
    
    # Verify the save
    embeddings, labels, paths = EmbeddingStorage.load(output_file)
    print("\nCurrent progress statistics:")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of images: {len(paths)}")
    print(f"Images per class: {pd.Series(labels).value_counts().sort_index()}")
    print("Progress saved successfully!")

def main(test_mode=True):
    # Create output directory
    os.makedirs('data', exist_ok=True)
    
    # Initialize embedding storage with appropriate filename
    output_file = 'data/imagenet_test_embeddings.h5' if test_mode else 'data/imagenet_embeddings.h5'
    storage = EmbeddingStorage(output_file)
    
    # Load image mapping
    mapping_file = '../data/imagenet-organized/imagenet_mapping.csv'
    if not os.path.exists(mapping_file):
        print(f"Error: Mapping file {mapping_file} not found!")
        print("Please run create_imagenet_mapping.py first.")
        sys.exit(1)
    
    image_df = pd.read_csv(mapping_file)
    imagenet_dir = '../data/imagenet-organized'
    
    if not os.path.exists(imagenet_dir):
        print(f"Error: Directory {imagenet_dir} not found!")
        sys.exit(1)
    
    if test_mode:
        print("Running in test mode with 350 images...")
        # Take 50 images from each superclass (indices 4-10)
        test_samples = []
        for class_idx in range(4, 11):
            class_data = image_df[image_df['superclass_index'] == class_idx].head(700)
            test_samples.append(class_data)
        image_df = pd.concat(test_samples)
    
    try:
        # Process images in chunks
        chunk_size = 500
        total_images = len(image_df)
        
        for chunk_start in range(0, total_images, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_images)
            chunk_df = image_df.iloc[chunk_start:chunk_end]
            
            print(f"\nProcessing chunk {chunk_start//chunk_size + 1} ({chunk_start}-{chunk_end})")
            
            # Get full paths for the chunk
            image_paths = [
                os.path.join(imagenet_dir, row['superclass_name'], row['subclass_index'], row['image'])
                for _, row in chunk_df.iterrows()
            ]
            
            # Process the chunk
            results = process_images_batch(image_paths, batch_size=20)
            
            # Add processed results to storage
            for (embedding, img_path), (_, row) in zip(results, chunk_df.iterrows()):
                storage.add_sample(
                    embedding=embedding,
                    label=row['superclass_index'],
                    image_path=img_path
                )
            
            # Save progress after each chunk
            save_progress(storage, output_file)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving current progress...")
        save_progress(storage, output_file)
        sys.exit(0)
    except Exception as e:
        print(f"\nEncountered error: {str(e)}")
        print("Saving current progress...")
        save_progress(storage, output_file)
        raise e

    print("\nProcessing completed successfully!")
    save_progress(storage, output_file)

if __name__ == "__main__":
    main(test_mode=True)  # Set to True for test run 