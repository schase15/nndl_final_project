import os
import argparse
import subprocess

SUPERCLASS_MAP = {0: 'bird', 1: 'dog', 2: 'reptile'}

parser = argparse.ArgumentParser()
parser.add_argument('--csv', required=True, help='CSV from predict_superclass (superclass_predictions.csv)')
parser.add_argument('--image_dir', required=True, help='Directory of images to classify')
parser.add_argument('--output_dir', required=True, help='Directory to save all outputs')
parser.add_argument('--dev', action='store_true', help='Run in dev mode (faster, fewer images)')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

for idx, class_name in SUPERCLASS_MAP.items():
    print(f'\nProcessing superclass: {class_name}')
    # 1. Run cosine_subclass_predict.py for this class
    cosine_cmd = [
        'python', os.path.join(os.path.dirname(__file__), 'cosine_subclass_predict.py'),
        '--superclass', str(idx),
        '--csv', args.csv,
        '--image_dir', args.image_dir,
        '--output_dir', args.output_dir
    ]
    if args.dev:
        cosine_cmd.append('--dev')
    print(f'Running cosine_subclass_predict for {class_name}...')
    subprocess.run(cosine_cmd, check=True)

    # 2. Run llm_subclass_classifier.py for this class
    subclass_csv = os.path.join(args.output_dir, 'subclass_predictions', f'{class_name}_subclass_predictions.csv')
    llm_cmd = [
        'python', os.path.join(os.path.dirname(__file__), 'llm_subclass_classifier.py'),
        '--superclass', class_name,
        '--csv', subclass_csv,
        '--image_dir', args.image_dir,
        '--output_csv', os.path.join(args.output_dir, f'{class_name}_subclass_predictions.csv')
    ]
    if args.dev:
        llm_cmd.append('--dev')
    print(f'Running llm_subclass_classifier for {class_name}...')
    subprocess.run(llm_cmd, check=True)
    print(f'Output CSV for {class_name}: {os.path.join(args.output_dir, f"{class_name}_subclass_predictions.csv")}')

print('\nSubclass prediction pipeline complete!') 