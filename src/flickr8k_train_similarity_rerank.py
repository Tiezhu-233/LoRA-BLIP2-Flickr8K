from flickr8k_dataset_full import Flickr8kDataset
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import os
import pandas as pd
from tqdm import tqdm
import time
import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from collections import defaultdict
import json
from sentence_transformers import SentenceTransformer, util
import logging
import re
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)  # Comment translated to English and cleaned.

# Comment translated to English and cleaned.
logger.info("")

# Comment translated to English and cleaned.
base_dir = os.getenv("PROJECT_ROOT", ".")

# Comment translated to English and cleaned.
image_dir = os.getenv("FLICKR8K_IMAGE_DIR", os.path.join(base_dir, "data", "flickr8k", "Flickr8k_Dataset"))
caption_path = os.getenv("FLICKR8K_CAPTION_FILE", os.path.join(base_dir, "data", "flickr8k", "captions.txt"))

# Comment translated to English and cleaned.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f": {device}")

# Comment translated to English and cleaned.
try:
    print("...")
    similarity_model = SentenceTransformer('./all-MiniLM-L6-v2').to(device)
    print("")
except Exception as e:
    print(f": {str(e)}")
    similarity_model = None

try:
    print("BLIP...")
    processor = Blip2Processor.from_pretrained("blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "blip2-opt-2.7b",
        torch_dtype=torch.float16
    ).to(device)
    print("BLIP")
except Exception as e:
    print(f": {str(e)}")
    exit(1)

# Comment translated to English and cleaned.
try:
    print("...")
    dataset = Flickr8kDataset(
        image_dir=image_dir,
        caption_path=caption_path,
        processor=processor,
        split='all'
    )
    print(f" {len(dataset)} ")
    
    # Comment translated to English and cleaned.
    print("...")
    sample = dataset[0]
    print(":", sample["pixel_values"].shape)
    print(":", list(sample.keys()))
    
except Exception as e:
    print(f": {str(e)}")
    exit(1)

# Comment translated to English and cleaned.
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "flickr8k_blip2_descriptions_similarity.csv")

# Comment translated to English and cleaned.
# Comment translated to English and cleaned.
reference_descriptions = defaultdict(list)

# Comment translated to English and cleaned.
print("\n...")
for i in tqdm(range(len(dataset)), desc=""):
    metadata = dataset.get_metadata(i)
    base_id = metadata["base_image_id"]
    caption = metadata["caption"]
    reference_descriptions[base_id].append(caption)

# Comment translated to English and cleaned.
print(f"\n...")
model.eval()

results = []
batch_size = 8
num_batches = (len(dataset) + batch_size - 1) // batch_size

start_time = time.time()
skipped_indices = []  # Comment translated to English and cleaned.

def select_most_similar(candidates, reference, similarity_model):
    """
    
    """
    if similarity_model is None:
        # Comment translated to English and cleaned.
        ref_len = len(reference.split())
        best_idx = min(range(len(candidates)), 
                       key=lambda i: abs(len(candidates[i].split()) - ref_len))
        return candidates[best_idx]
    
    # Comment translated to English and cleaned.
    candidate_embeddings = similarity_model.encode(candidates, convert_to_tensor=True)
    ref_embedding = similarity_model.encode([reference], convert_to_tensor=True)
    
    # Comment translated to English and cleaned.
    similarities = util.pytorch_cos_sim(ref_embedding, candidate_embeddings)[0]
    
    # Comment translated to English and cleaned.
    best_idx = torch.argmax(similarities).item()
    return candidates[best_idx]

def process_generated_text(text,prompt):
    """"""
    text = re.sub(f"^{re.escape(prompt)}", "", text).strip()
    return text 

with torch.no_grad():
    progress_bar = tqdm(range(num_batches), desc="")
    for i in progress_bar:
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(dataset))
        
        # Comment translated to English and cleaned.
        batch_samples = []
        valid_indices = []
        metadata_list = []  # Comment translated to English and cleaned.
        
        for j in range(start_idx, end_idx):
            try:
                # Comment translated to English and cleaned.
                sample = dataset[j]
                
                # Comment translated to English and cleaned.
                metadata = dataset.get_metadata(j)
                
                batch_samples.append(sample)
                valid_indices.append(j)
                metadata_list.append(metadata)
            except Exception as e:
                print(f"\n {j}: {str(e)}")
                skipped_indices.append(j)
        
        if not batch_samples:
            continue
        
        # Comment translated to English and cleaned.
        pixel_values = torch.cat([s["pixel_values"] for s in batch_samples]).to(device)
        
        # Comment translated to English and cleaned.
        input_ids_list = [s["input_ids"] for s in batch_samples]
        attention_mask_list = [s["attention_mask"] for s in batch_samples]
        
        # Comment translated to English and cleaned.
        max_length = max([ids.shape[1] for ids in input_ids_list])
        
        # Comment translated to English and cleaned.
        padded_input_ids = []
        padded_attention_mask = []
        
        for input_ids, attn_mask in zip(input_ids_list, attention_mask_list):
            # Comment translated to English and cleaned.
            seq_len = input_ids.shape[1]
            
            # Comment translated to English and cleaned.
            pad_len = max_length - seq_len
            
            # Comment translated to English and cleaned.
            padded_input = torch.cat([
                input_ids,
                torch.full((1, pad_len), processor.tokenizer.pad_token_id, device=input_ids.device)
            ], dim=1)
            
            # Comment translated to English and cleaned.
            padded_attn = torch.cat([
                attn_mask,
                torch.zeros((1, pad_len), device=attn_mask.device)
            ], dim=1)
            
            padded_input_ids.append(padded_input)
            padded_attention_mask.append(padded_attn)
            
        # Comment translated to English and cleaned.
        input_ids = torch.cat(padded_input_ids).to(device)
        attention_mask = torch.cat(padded_attention_mask).to(device)
        
        # Comment translated to English and cleaned.
       
        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            do_sample=False,  # Comment translated to English and cleaned.
            num_return_sequences=3,  # Comment translated to English and cleaned.
            attention_mask=attention_mask,
            max_new_tokens=80,
            num_beams=5,
            temperature=0.7, 
            early_stopping=True,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            eos_token_id=model.config.eos_token_id,
        )
        
        # Comment translated to English and cleaned.
        all_generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Comment translated to English and cleaned.
        batch_size_actual = len(batch_samples)
        grouped_texts = [all_generated_texts[i*3 : (i+1)*3] for i in range(batch_size_actual)]
        
                # Comment translated to English and cleaned.
        for idx, (candidates, metadata) in enumerate(zip(grouped_texts, metadata_list)):
            original_caption = metadata["caption"]

            prompt = ('Identify which objects are present in the diagram and specify how they are related. Based on these findings, please generate a 15-30 word description for this image.')
            # Comment translated to English and cleaned.
            processed_candidates = [process_generated_text(text,prompt) for text in candidates]
            
            # Comment translated to English and cleaned.
            best_caption = select_most_similar(
                candidates=processed_candidates,
                reference=original_caption,
                similarity_model=similarity_model
            )

            # Comment translated to English and cleaned.
            logger.info(
                f" {valid_indices[idx]} | "
                f": \"{original_caption}\" | "
                f": \"{best_caption}\""
            )

            # Comment translated to English and cleaned.
            results.append({
                "full_image_id": metadata["full_image_id"],
                "base_image_id": metadata["base_image_id"],
                "generated_caption": best_caption,
                "original_caption": original_caption,
                "all_candidates": processed_candidates,
                "dataset_index": valid_indices[idx]
            })

            
        # Comment translated to English and cleaned.
        progress_bar.set_postfix({
            "": f"{len(results)}/{len(dataset)}",
            "": len(skipped_indices)
        })

# Comment translated to English and cleaned.
if results:
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
if results:
    print("\n5:")
    for i, res in enumerate(results[:5]):
        print(f"\n {i+1}:")
        print(f"  : {res['original_caption']}")
        print(f"  : {res['generated_caption']}")
        print(f"  : {res['all_candidates']}")
else:
    print("\n")

