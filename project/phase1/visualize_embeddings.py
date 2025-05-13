"""
This script visualizes the training embeddings clustered by superclass name.
It uses t-SNE (or PCA if t-SNE is too slow) to reduce dimensionality and matplotlib for plotting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from embedding_storage import EmbeddingStorage
import seaborn as sns
import os

# Paths to data
IMAGENET_EMBEDDINGS_PATH = 'data/train_imagenet_embeddings_700.h5'
NNDL_EMBEDDINGS_PATH = 'data/train_embeddings_nndl_700.h5'
IMAGENET_MAPPING_PATH = '../data/imagenet-organized/imagenet_mapping.csv'
NNDL_MAPPING_PATH = '../data/Released_Data_NNDL_2025/train_data.csv'

# Load embeddings and mapping files
imagenet_embeddings, imagenet_labels, imagenet_paths = EmbeddingStorage.load(IMAGENET_EMBEDDINGS_PATH)
nndl_embeddings, nndl_labels, nndl_paths = EmbeddingStorage.load(NNDL_EMBEDDINGS_PATH)
imagenet_map = pd.read_csv(IMAGENET_MAPPING_PATH)
nndl_map = pd.read_csv(NNDL_MAPPING_PATH)

# Combine all data for visualization
all_embeddings = []
all_class_names = []

# ImageNet
for emb, path in zip(imagenet_embeddings, imagenet_paths):
    fname = os.path.basename(path)
    row = imagenet_map[imagenet_map['image'] == fname]
    if not row.empty:
        class_name = row.iloc[0]['superclass_name']
        all_embeddings.append(emb)
        all_class_names.append(class_name)

# NNDL
for emb, path in zip(nndl_embeddings, nndl_paths):
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
        all_class_names.append(class_name)

all_embeddings = np.array(all_embeddings)
all_class_names = np.array(all_class_names)

# Dimensionality reduction
print("Running t-SNE (this may take a while)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(all_embeddings)

# Plot
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    hue=all_class_names,
    palette='tab10',
    alpha=0.7,
    s=30
)
plt.title('t-SNE Visualization of Training Embeddings by Superclass')
plt.legend(title='Superclass', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('results/embedding_tsne.png', dpi=200)
plt.show() 