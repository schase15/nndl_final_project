"""
This script creates a mapping CSV file from the imagenet-organized folder structure.
Format: imagenet-organized/<superclass>/<subclass_index>/images.JPEG
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

def get_superclass_mapping() -> Dict[str, int]:
    """
    Returns the mapping of folder names to superclass indices.
    Indices start at 4 to avoid overlap with NNDL dataset (0-2).
    """
    return {
        'Arthropod': 4,
        'Bear': 5,
        'Bovid': 6,
        'Cats': 7,
        'Elephants': 8,
        'Insects': 9,
        'Primates': 10
    }

def create_image_mapping(imagenet_dir: str) -> List[Tuple[str, int, int, str]]:
    """
    Creates a list of (image_filename, superclass_index, subclass_index, superclass_name) tuples
    from the imagenet directory structure.
    """
    superclass_mapping = get_superclass_mapping()
    image_data = []
    
    # Walk through the imagenet directory
    for superclass_folder in os.listdir(imagenet_dir):
        superclass_path = os.path.join(imagenet_dir, superclass_folder)
        # print(superclass_path)
        # breakpoint()
        
        # Skip if not a directory or not a superclass
        if not os.path.isdir(superclass_path) or superclass_folder not in superclass_mapping:
            continue
        
        superclass_idx = superclass_mapping[superclass_folder]
        # print(superclass_idx)
        # breakpoint()
        
        # Process each subclass folder
        for subclass_folder in os.listdir(superclass_path):
            subclass_path = os.path.join(superclass_path, subclass_folder)
            # print(subclass_path)
            # breakpoint()
            
            # Skip if not a directory
            if not os.path.isdir(subclass_path):
                continue
                
            try:
                # Process all images in this subclass
                for file in os.listdir(subclass_path):
                    if file.lower().endswith(('.jpg', '.jpeg')):
                        image_data.append((file, superclass_idx, subclass_folder, superclass_folder))
                        # print(file, superclass_idx, subclass_folder, superclass_folder)
                        # breakpoint()
            except ValueError:
                print(f"Warning: Invalid subclass folder name: {subclass_folder} in {superclass_folder}")
                continue
    
    return image_data

def main():
    # Define paths
    imagenet_dir = '../data/imagenet-organized'
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(imagenet_dir):
        print(f"Error: Directory {imagenet_dir} not found!")
        return
    
    print("Creating image mapping from imagenet directory...")
    print(f"Using directory: {os.path.abspath(imagenet_dir)}")
    
    # Create the mapping
    image_data = create_image_mapping(imagenet_dir)
    
    if not image_data:
        print("No images found in the superclass directories!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(image_data, columns=['image', 'superclass_index', 'subclass_index', 'superclass_name'])
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'imagenet_mapping.csv')
    df.to_csv(output_file, index=False)
    
    # Print statistics
    print("\nMapping created successfully!")
    print(f"Total images: {len(df)}")
    print("\nImages per superclass:")
    for name, group in df.groupby('superclass_name'):
        n_subclasses = len(group['subclass_index'].unique())
        print(f"{name}: {len(group)} images, {n_subclasses} subclasses (superclass index: {group['superclass_index'].iloc[0]})")
    print(f"\nMapping saved to: {output_file}")

if __name__ == "__main__":
    main() 