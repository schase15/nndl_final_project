import os
import argparse
import pandas as pd
import base64
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm

NOVEL_SUBCLASS_IDX = 87

async def get_llm_classification(image_path, system_prompt, client):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    response = await client.responses.create(
        model="gpt-4.1",
        instructions=system_prompt,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Classify this image as one of the known subclasses index (integer) from the mapping above, or 87 for novel. Respond with the subclass index or 87 for novel."},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                ]
            }
        ]
    )
    return response.output_text.strip()

async def classify_one(idx, image_path, system_prompt, client, semaphore):
    async with semaphore:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return idx, 'unsure'
        try:
            pred = await get_llm_classification(image_path, system_prompt, client)
            # Try to parse as int, else return 'unsure'
            try:
                pred_int = int(pred)
                return idx, pred_int
            except Exception:
                if pred.lower() == 'novel':
                    return idx, NOVEL_SUBCLASS_IDX
                return idx, 'unsure'
        except Exception as e:
            print(f"Error for {image_path}: {e}")
            return idx, 'unsure'

async def main_async(df, unsure_indices, image_dir, system_prompt, concurrency=10):
    phase2_preds = [None] * len(df)
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    for idx in unsure_indices:
        row = df.iloc[idx]
        image_path = os.path.join(image_dir, row['image_id'])
        tasks.append(classify_one(idx, image_path, system_prompt, client, semaphore))
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='LLM subclassifying unsure images'):
        idx, pred = await fut
        phase2_preds[idx] = pred
    return phase2_preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--superclass', required=True, choices=['dog', 'bird', 'reptile'])
    parser.add_argument('--csv', required=True, help='CSV from cosine_subclass_predict')
    parser.add_argument('--image_dir', required=True, help='Directory of images to classify')
    parser.add_argument('--output_csv', required=True, help='Output CSV with LLM subclass predictions')
    parser.add_argument('--dev', action='store_true', help='Dev mode: only process 10 unsure images')
    args = parser.parse_args()

    # Load the subclass mapping
    subclass_map_df = pd.read_csv('../data/Released_Data_NNDL_2025/subclass_mapping.csv')
    train_df = pd.read_csv('../data/Released_Data_NNDL_2025/train_data.csv')

    # Create a mapping from superclass_index to list of subclass_index
    superclass_to_subclass_indices = train_df.groupby('superclass_index')['subclass_index'].unique().apply(list).to_dict()

    # Map from superclass name to index
    superclass_name_to_index = {'bird': 0, 'dog': 1, 'reptile': 2}
    superclass_idx = superclass_name_to_index[args.superclass]
    
    # Get the list of subclass indices for this superclass
    subclass_indices = superclass_to_subclass_indices[superclass_idx]
    
    # Build mapping from subclass index to name for this superclass
    subclass_map = {int(row['index']): row['class'] for _, row in subclass_map_df.iterrows() if int(row['index']) in subclass_indices}
    
    # Format mapping for prompt
    mapping_str = '\n'.join([f"{idx}: {name}" for idx, name in subclass_map.items() if idx != 87])

    system_prompt = f"""
You are an expert at identifying {args.superclass} breeds (subclasses). You will be shown an image of a {args.superclass}, and you must classify it as one of the following subclasses (by index), or as novel if it is not any of these:

Consider an image from all angles and perspectives. It may be a close up, a side view, or a back view. Or the image may be upside down.

Subclass mapping:
{mapping_str}

Your task:
- Return the integer index of the subclass (from the mapping above) if you are confident.
- If the image is not any of the above subclasses, return 87 for novel.
- Do not return anything else, just the integer.
"""

    df = pd.read_csv(args.csv)
    unsure_mask = df['phase1_pred_subclass'] == 'unsure'
    unsure_indices = df[unsure_mask].index.tolist()
    if args.dev:
        unsure_indices = unsure_indices[:10]
        print(f"[DEV MODE] Only processing {len(unsure_indices)} unsure images.")

    # Run async LLM classification
    phase2_preds = asyncio.run(main_async(df, unsure_indices, args.image_dir, system_prompt, concurrency=50))

    df['phase2_pred_subclass'] = phase2_preds

    # Add final_subclass_pred column
    df['final_subclass_pred'] = df['phase2_pred_subclass'].fillna(df['phase1_pred_subclass'])

    # convert final_subclass_pred to int
    df['final_subclass_pred'] = df['final_subclass_pred'].astype(int)
    
    df.to_csv(args.output_csv, index=False)
    print(f"Saved LLM subclass predictions to {args.output_csv}")

if __name__ == "__main__":
    main() 