"""
This script calculates centroids for the following classes using 70% of the embeddings for each class:
Arthropod, Bear, Bovid, Insects, Dog, Bird, Reptile.
Only saves the class centroids.
"""

import os
import numpy as np
import pandas as pd
from embedding_storage import EmbeddingStorage
from sklearn.model_selection import train_test_split

# Classes to calculate centroids for
CLASSES = ['Arthropod', 'Bear', 'Bovid', 'Insects', 'Dog', 'Bird', 'Reptile']

# Mapping from class name to index (update as needed)
CLASS_NAME_TO_INDEX = {
    'Arthropod': 4,
    'Bear': 5,
    'Bovid': 6,
    'Insects': 9,
    'Dog': 1,
    'Bird': 0,
    'Reptile': 2
}

# Path to embeddings file and mapping file
IMAGENET_EMBEDDINGS_PATH = 'data/train_imagenet_embeddings_700.h5'
NNDL_EMBEDDINGS_PATH = 'data/train_embeddings_nndl_700.h5'
IMAGENET_MAPPING_PATH = '../data/imagenet-organized/imagenet_mapping.csv'
NNDL_MAPPING_PATH = '../data/Released_Data_NNDL_2025/train_data.csv'

# Output file
CENTROIDS_OUTPUT = 'data/class_centroids.npy'

# Load all embeddings and labels from both datasets
imagenet_embeddings, imagenet_labels, imagenet_paths = EmbeddingStorage.load(IMAGENET_EMBEDDINGS_PATH)
nndl_embeddings, nndl_labels, nndl_paths = EmbeddingStorage.load(NNDL_EMBEDDINGS_PATH)

# Load mapping files for class name lookup
imagenet_map = pd.read_csv(IMAGENET_MAPPING_PATH)
nndl_map = pd.read_csv(NNDL_MAPPING_PATH)

# Combine all data into a single DataFrame for easier processing
all_embeddings = []
all_labels = []
all_class_names = []

# Process ImageNet data
for emb, label, path in zip(imagenet_embeddings, imagenet_labels, imagenet_paths):
    fname = os.path.basename(path)
    row = imagenet_map[imagenet_map['image'] == fname]
    if not row.empty:
        class_name = row.iloc[0]['superclass_name']
        all_embeddings.append(emb)
        all_labels.append(label)
        all_class_names.append(class_name)

# Process NNDL data
for emb, label, path in zip(nndl_embeddings, nndl_labels, nndl_paths):
    fname = os.path.basename(path)
    row = nndl_map[nndl_map['image'] == fname]
    if not row.empty:
        idx = int(row.iloc[0]['superclass_index'])
        if idx == 0:
            class_name = 'Bird'
        elif idx == 1:
            class_name = 'Dog'
        elif idx == 2:
            class_name = 'Reptile'
        else:
            continue
        all_embeddings.append(emb)
        all_labels.append(label)
        all_class_names.append(class_name)

all_embeddings = np.array(all_embeddings)
all_labels = np.array(all_labels)
all_class_names = np.array(all_class_names)

# Calculate centroids using 70% of the data for each class
centroids = {}
for class_name in CLASSES:
    class_mask = all_class_names == class_name
    class_embs = all_embeddings[class_mask]
    if len(class_embs) == 0:
        print(f"Warning: No embeddings found for class {class_name}")
        continue
    # Use 70% of the data for centroid calculation
    train_embs, _ = train_test_split(class_embs, test_size=0.3, random_state=42)
    centroid = np.mean(train_embs, axis=0)
    centroids[class_name] = centroid
    print(f"Class {class_name}: {len(train_embs)} used for centroid, centroid shape: {centroid.shape}")

# Save centroids
np.save(CENTROIDS_OUTPUT, centroids)
print(f"\nCentroids saved to {CENTROIDS_OUTPUT}") 