import os
import argparse
import subprocess
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', required=True, help='Directory of images to classify')
parser.add_argument('--output_dir', required=True, help='Directory to save all outputs')
parser.add_argument('--dev', action='store_true', help='Run in dev mode (faster, fewer images)')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# 1. Run cosine_superclass_predict.py
cosine_cmd = [
    'python', os.path.join(os.path.dirname(__file__), 'cosine_superclass_predict.py'),
    '--image_folder', args.image_dir,
    '--output_dir', args.output_dir
]
if args.dev:
    cosine_cmd.append('--dev')
print('Running cosine_superclass_predict...')
# subprocess.run(cosine_cmd, check=True)

# 2. Run llm_superclass_classifier.py
csv_path = os.path.join(args.output_dir, 'superclass_predictions.csv')
llm_cmd = [
    'python', os.path.join(os.path.dirname(__file__), 'llm_superclass_classifier.py'),
    '--csv', csv_path,
    '--image_dir', args.image_dir
]
if args.dev:
    llm_cmd.append('--dev')
print('Running llm_superclass_classifier...')
subprocess.run(llm_cmd, check=True)

# 3. Run create_superclass_subsets_from_csv.py
all_embeddings_path = os.path.join(args.output_dir, 'all_embeddings.h5')
subsets_dir = os.path.join(args.output_dir, 'superclass_subsets')
os.makedirs(subsets_dir, exist_ok=True)
subset_cmd = [
    'python', os.path.join(os.path.dirname(__file__), 'create_superclass_subsets_from_csv.py'),
    '--csv', csv_path,
    '--embeddings', all_embeddings_path,
    '--output_dir', subsets_dir
]
print('Running create_superclass_subsets_from_csv...')
subprocess.run(subset_cmd, check=True)

print('\nSuperclass pipeline complete!')
print(f'CSV: {csv_path}')
print(f'All embeddings: {all_embeddings_path}')
print(f'Subset HDF5 files in: {subsets_dir}') 