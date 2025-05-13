import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

VAL_RUN_DIR = os.path.dirname(__file__)
LABELS_CSV = os.path.join(VAL_RUN_DIR, 'image_labels.csv')
IMAGES_DIR = os.path.join(VAL_RUN_DIR, 'val_images')
SUBSET_CSV = os.path.join(VAL_RUN_DIR, 'image_labels_subset.csv')
SUBSET_IMAGES_DIR = os.path.join(VAL_RUN_DIR, 'val_images_subset')
MAX_SAMPLES = 2000

# Read the labels
labels_df = pd.read_csv(LABELS_CSV)

# If there are more than MAX_SAMPLES, stratify by both superclass and subclass
if len(labels_df) > MAX_SAMPLES:
    # Create a stratification key
    labels_df['stratify_key'] = labels_df['superclass_index'].astype(str) + '_' + labels_df['subclass_index'].astype(str)
    # If there are too many unique keys, fallback to superclass only
    n_unique = labels_df['stratify_key'].nunique()
    if n_unique > MAX_SAMPLES:
        stratify_col = 'superclass_index'
    else:
        stratify_col = 'stratify_key'
    subset_df, _ = train_test_split(
        labels_df,
        train_size=MAX_SAMPLES,
        stratify=labels_df[stratify_col],
        random_state=42
    )
    subset_df = subset_df.drop(columns=['stratify_key'], errors='ignore')
else:
    subset_df = labels_df.copy()

# Make output dir
os.makedirs(SUBSET_IMAGES_DIR, exist_ok=True)

# Copy images
for fname in subset_df['image']:
    src = os.path.join(IMAGES_DIR, fname)
    dst = os.path.join(SUBSET_IMAGES_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)

# Write new CSV
subset_df.to_csv(SUBSET_CSV, index=False)

# Print summary
grouped = subset_df.groupby(['superclass_index', 'subclass_index']).size().reset_index(name='count')
print(f"Subset created with {len(subset_df)} images.")
print(grouped) 