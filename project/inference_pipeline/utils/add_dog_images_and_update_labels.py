import os
import random
import shutil
import pandas as pd

DOG_SRC = os.path.join('../data', 'extra_target_images', 'Dog')
VAL_IMAGES = os.path.join('val_run', 'val_images')
LABELS_CSV = os.path.join('val_run', 'image_labels.csv')
SUPERCLASS_INDEX = 1
SUBCLASS_INDEX = 87
N_PER_SUBFOLDER = 200

# 1. Sample 200 images from each subfolder
subfolders = [os.path.join(DOG_SRC, d) for d in os.listdir(DOG_SRC) if os.path.isdir(os.path.join(DOG_SRC, d))]
new_rows = []
existing_filenames = set(os.listdir(VAL_IMAGES))

for subfolder in subfolders:
    all_images = [f for f in os.listdir(subfolder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sampled = random.sample(all_images, min(N_PER_SUBFOLDER, len(all_images)))
    for fname in sampled:
        src_path = os.path.join(subfolder, fname)
        # Avoid filename collision by prefixing with subfolder if needed
        out_fname = fname
        if out_fname in existing_filenames:
            out_fname = f"{os.path.basename(subfolder)}_{fname}"
        dst_path = os.path.join(VAL_IMAGES, out_fname)
        shutil.copy2(src_path, dst_path)
        new_rows.append({'image': out_fname, 'superclass_index': SUPERCLASS_INDEX, 'subclass_index': SUBCLASS_INDEX})
        existing_filenames.add(out_fname)

# 2. Append new rows to image_labels.csv
labels_df = pd.read_csv(LABELS_CSV)
labels_df = pd.concat([labels_df, pd.DataFrame(new_rows)], ignore_index=True)

# 3. Edit subclass_index: if it starts with n0, set to 87
# (Assume subclass_index is int or str, so convert to str for check)
def fix_subclass(val):
    val_str = str(val)
    if val_str.startswith('n0'):
        return SUBCLASS_INDEX
    return val
labels_df['subclass_index'] = labels_df['subclass_index'].apply(fix_subclass)

labels_df.to_csv(LABELS_CSV, index=False)

print(f"Added {len(new_rows)} images from {len(subfolders)} subfolders.")
print(f"image_labels.csv now has {len(labels_df)} rows.") 