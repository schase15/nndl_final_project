import h5py
import numpy as np
from pathlib import Path

class EmbeddingStorage:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.embeddings = []
        self.labels = []
        self.image_paths = []
    
    def add_sample(self, embedding: np.ndarray, label: int, image_path: str):
        """Add a single sample to the storage."""
        self.embeddings.append(embedding)
        self.labels.append(label)
        self.image_paths.append(image_path)
    
    def save(self):
        """Save all data to HDF5 file."""
        with h5py.File(self.output_path, 'w') as f:
            f.create_dataset('embeddings', data=np.array(self.embeddings))
            f.create_dataset('labels', data=np.array(self.labels))
            # Store image paths as ASCII strings
            image_paths_ascii = [str(p).encode('ascii') for p in self.image_paths]
            f.create_dataset('image_paths', data=image_paths_ascii)
    
    @staticmethod
    def load(file_path: str):
        """Load data from HDF5 file."""
        with h5py.File(file_path, 'r') as f:
            embeddings = f['embeddings'][:]
            labels = f['labels'][:]
            image_paths = [p.decode('ascii') for p in f['image_paths'][:]]
        return embeddings, labels, image_paths
