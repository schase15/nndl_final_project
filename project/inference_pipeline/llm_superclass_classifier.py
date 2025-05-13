import os
import sys
import argparse
import pandas as pd
import base64
from openai import AsyncOpenAI
from tqdm import tqdm
import asyncio
import aiofiles

LLM_PROMPT = """
You are an expert at identifying images of birds, dogs and reptiles. 
You will be shown images and you must confidently classify them as one of the following:
- Bird
- Dog
- Reptile
- Novel 

You will most likely see more images that are not birds, dogs, or reptiles, these are to be classificed as 'novel'

Consider an image from all angles and perspectives. It may be a close up, a side view, or a back view. Or the image may be upside down.

Your task is to make the final classification decision. Consider these guidelines:
1. Consider typical features:
   - Birds: feathers, beaks, wings
   - Dogs: fur, typical canine features, domestic dog breeds
   - Reptiles: scales, cold-blooded features, typical reptilian anatomy
2. If the image could be a different type of animal other than bird, dog, or reptile, classify as 'novel'
3. Double check your answer before responding

There may be background noise or other objects in the image that are not the animal itself. Focus on classifying the animal itself.

Respond with ONLY ONE WORD: 'bird', 'dog', 'reptile', or 'novel'
"""

LABEL_MAP = {'bird': int(0), 'dog': int(1), 'reptile': int(2), 'novel': int(3)}

client = AsyncOpenAI()

async def encode_image(image_path):
    async with aiofiles.open(image_path, "rb") as image_file:
        return base64.b64encode(await image_file.read()).decode('utf-8')

def encode_image_sync(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def get_llm_classification(image_path):
    base64_image = encode_image_sync(image_path)
    response = await client.responses.create(
        model="gpt-4.1",
        instructions=LLM_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Classify this image as either 'bird', 'dog', 'reptile', or 'novel'."},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                ]
            }
        ]
    )
    return response.output_text.strip().lower()

async def get_llm_classification_with_retries(image_path, max_retries=5):
    valid_labels = {'bird', 'dog', 'reptile', 'novel'}
    for attempt in range(1, max_retries + 1):
        try:
            llm_pred = await get_llm_classification(image_path)
            llm_pred = llm_pred.strip().lower()
            if llm_pred in valid_labels:
                return llm_pred
            else:
                print(f"[Attempt {attempt}] Invalid LLM output for {image_path}: '{llm_pred}'. Retrying...")
        except Exception as e:
            print(f"[Attempt {attempt}] Error classifying image {image_path}: {e}. Retrying...")
    print(f"[ERROR] Max retries reached for {image_path}. Classifying as 'novel'.")
    return 'novel'

async def classify_one(idx, image_path, semaphore):
    async with semaphore:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return idx, 'novel'
        pred = await get_llm_classification_with_retries(image_path)
        return idx, pred

async def main_async(df, unsure_indices, image_dir, concurrency=10):
    phase2_preds = [None] * len(df)
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    for idx in unsure_indices:
        row = df.iloc[idx]
        image_path = os.path.join(image_dir, row['image_id'])
        tasks.append(classify_one(idx, image_path, semaphore))
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='LLM classifying unsure images'):
        idx, pred = await fut
        phase2_preds[idx] = LABEL_MAP[pred]
    return phase2_preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='CSV from cosine_superclass_predict')
    parser.add_argument('--image_dir', required=True, help='Folder containing images')
    parser.add_argument('--dev', action='store_true', help='Dev mode: only process 10 unsure images')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    unsure_mask = df['phase1_pred_superclass'] == 'unsure'
    unsure_indices = df[unsure_mask].index.tolist()
    if args.dev:
        unsure_indices = unsure_indices[:10]
        print(f"[DEV MODE] Only processing {len(unsure_indices)} unsure images.")

    # Run async LLM classification
    phase2_preds = asyncio.run(main_async(df, unsure_indices, args.image_dir, concurrency=50))

    df['phase2_pred_superclass'] = phase2_preds

    # Add final_superclass_pred column
    df['final_superclass_pred'] = df['phase2_pred_superclass'].fillna(df['phase1_pred_superclass'])


    df.to_csv(args.csv, index=False)
    print(f"Saved LLM predictions to {args.csv}")
    print(df['phase2_pred_superclass'].value_counts(dropna=False))



if __name__ == "__main__":
    main() 