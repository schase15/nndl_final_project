import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

# Add phase1 to sys.path for EmbeddingStorage
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'phase1')))
from embedding_storage import EmbeddingStorage

SUPERCLASSES = ['dog', 'bird', 'reptile']

for superclass in SUPERCLASSES:
    h5_path = f'train_{superclass}.h5'
    out_dir = superclass
    os.makedirs(out_dir, exist_ok=True)
    embeddings, labels, image_paths = EmbeddingStorage.load(h5_path)
    labels = np.array(labels)
    embeddings = np.array(embeddings)
    # Group indices by subclass
    subclass_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        subclass_to_indices[label].append(idx)
    for subclass, indices in subclass_to_indices.items():
        # Sort indices for reproducibility
        indices = sorted(indices)
        n = len(indices)
        n_centroid = n // 2
        centroid_indices = indices[:n_centroid]
        centroid_embeddings = embeddings[centroid_indices]
        centroid = np.mean(centroid_embeddings, axis=0)
        out_path = os.path.join(out_dir, f'{subclass}_centroid.npy')
        np.save(out_path, centroid)
        print(f'Saved centroid for subclass {subclass} of {superclass} to {out_path}') 