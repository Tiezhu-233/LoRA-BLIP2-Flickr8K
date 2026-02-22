# Comment translated to English and cleaned.
PROMPT_TEMPLATES = [
    "Question: what is shown in this image? Answer:",
    "Describe the content of this photo:",
    "What can you see in this picture?",
    "Please describe what is happening in this image:",
    "Provide a detailed caption for this photograph:",
    "What is the main subject of this image?",
    "Explain the visual elements in this picture:",
    "Write a descriptive caption for this photo:",
    "What does this image depict?",
    "Describe the scene shown in this image:",
    "Provide a concise description of this photo:",
    "What is the key content of this picture?",
    "Describe the main elements in this image:",
    "What story does this image tell?",
    "Generate a caption for this photo:",
    "What is the central theme of this image?",
    "Describe the visual content of this picture:",
    "What can be observed in this photograph?",
    "Write an informative caption for this image:",
    "Describe the subject matter of this photo:"
]

from flickr8k_dataset_simple import Flickr8kDataset
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import os
import pandas as pd
from tqdm import tqdm
import time
import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge
import json

# Comment translated to English and cleaned.
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Comment translated to English and cleaned.
base_dir = os.getenv("PROJECT_ROOT", ".")

# Comment translated to English and cleaned.
image_dir = os.getenv("FLICKR8K_IMAGE_DIR", os.path.join(base_dir, "data", "flickr8k", "Flickr8k_Dataset"))
caption_path = os.getenv("FLICKR8K_CAPTION_FILE", os.path.join(base_dir, "data", "flickr8k", "captions.txt"))

# Comment translated to English and cleaned.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f": {device}")

try:
    print("...")
    processor = Blip2Processor.from_pretrained("blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "blip2-opt-2.7b",
        torch_dtype=torch.float16
    ).to(device)
    print("")
except Exception as e:
    print(f": {str(e)}")
    exit(1)

# Comment translated to English and cleaned.
try:
    print("...")
    dataset = Flickr8kDataset(
        image_dir=image_dir,
        caption_path=caption_path,
        processor=processor
    )
    print(f" {len(dataset)} ")
    
    # Comment translated to English and cleaned.
    print("...")
    sample = dataset[0]
    print(":", list(sample.keys()))
    
except Exception as e:
    print(f": {str(e)}")
    exit(1)

# Comment translated to English and cleaned.
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "flickr8k_blip2_descriptions.csv")
metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
prompt_stats_file = os.path.join(output_dir, "prompt_usage_stats.json")

# Comment translated to English and cleaned.
print(f"\n...")
print(f" {len(PROMPT_TEMPLATES)} ")
model.eval()

results = []
batch_size = 8
num_batches = (len(dataset) + batch_size - 1) // batch_size

start_time = time.time()
skipped_indices = []  # Comment translated to English and cleaned.

# Comment translated to English and cleaned.
prompt_usage = {prompt: 0 for prompt in PROMPT_TEMPLATES}

with torch.no_grad():
    progress_bar = tqdm(range(num_batches), desc="")
    for i in progress_bar:
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(dataset))
        
        # Comment translated to English and cleaned.
        batch_images = []
        valid_indices = []
        captions = []  # Comment translated to English and cleaned.
        image_ids = []  # Comment translated to English and cleaned.
        prompt_list = []  # Comment translated to English and cleaned.
        
        for j in range(start_idx, end_idx):
            try:
                # Comment translated to English and cleaned.
                sample = dataset[j]
                batch_images.append(sample["image"])
                valid_indices.append(j)
                image_ids.append(sample["image_id"])
                
                # Comment translated to English and cleaned.
                row = dataset.captions.iloc[j]
                captions.append(row["caption"])
                
                # Comment translated to English and cleaned.
                selected_prompt = random.choice(PROMPT_TEMPLATES)
                prompt_list.append(selected_prompt)
                prompt_usage[selected_prompt] += 1
                
            except Exception as e:
                print(f"\n {j}: {str(e)}")
                skipped_indices.append(j)
        
        if not batch_images:
            continue
        
        # Comment translated to English and cleaned.
        inputs = processor(
            images=batch_images,
            text=prompt_list,
            padding="max_length",
            max_length=40,  # Comment translated to English and cleaned.
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Comment translated to English and cleaned.
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            num_beams=3,
            early_stopping=True
        )
        
        # Comment translated to English and cleaned.
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Comment translated to English and cleaned.
        for idx, (text, caption, img_id, prompt) in enumerate(zip(generated_texts, captions, image_ids, prompt_list)):
            # Comment translated to English and cleaned.
            if prompt in text:
                text = text.replace(prompt, "").strip()
            
            # Comment translated to English and cleaned.
            results.append({
                "image_id": img_id,
                "generated_caption": text,
                "original_caption": caption,
                "prompt_used": prompt,
                "dataset_index": valid_indices[idx]
            })
            
        # Comment translated to English and cleaned.
        progress_bar.set_postfix({
            "": f"{len(results)}/{len(dataset)}",
            "": len(skipped_indices)
        })

# Comment translated to English and cleaned.
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)

# Comment translated to English and cleaned.
elapsed_time = time.time() - start_time
processed_count = len(results)
if processed_count > 0:
    images_per_sec = processed_count / elapsed_time
else:
    images_per_sec = 0

print("\n!")
print(f": {len(dataset)}")
print(f": {processed_count}")
print(f": {len(skipped_indices)}")
print(f": {elapsed_time:.2f} ")
print(f": {images_per_sec:.2f} /")
print(f": {output_file}")

# Comment translated to English and cleaned.
if skipped_indices:
    skipped_file = os.path.join(output_dir, "skipped_indices.txt")
    with open(skipped_file, 'w') as f:
        for idx in skipped_indices:
            f.write(f"{idx}\n")
    print(f": {skipped_file}")

# Comment translated to English and cleaned.
with open(prompt_stats_file, 'w') as f:
    json.dump(prompt_usage, f, indent=2)
print(f": {prompt_stats_file}")

# Comment translated to English and cleaned.
print("\n:")
for prompt, count in sorted(prompt_usage.items(), key=lambda x: x[1], reverse=True):
    print(f"{count}: {prompt}")

# Comment translated to English and cleaned.
if not results_df.empty:
    print("\n5:")
    print(results_df[["image_id", "prompt_used", "generated_caption"]].head())
