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
output_file = os.path.join(output_dir, "flickr8k_blip2_descriptions.csv")
metrics_file = os.path.join(output_dir, "evaluation_metrics.json")

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

# Comment translated to English and cleaned.
evaluation_data = {
    "hypotheses": [],  # Comment translated to English and cleaned.
    "references": []  # Comment translated to English and cleaned.
}

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
            attention_mask=attention_mask,
            max_new_tokens=50,
            num_beams=3,
            early_stopping=True
        )
        
        # Comment translated to English and cleaned.
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Comment translated to English and cleaned.
        for idx, (text, metadata) in enumerate(zip(generated_texts, metadata_list)):
            # Comment translated to English and cleaned.
            if "Question:" in text and "Answer:" in text:
                text = text.split("Answer:")[1].strip()
            
            # Comment translated to English and cleaned.
            results.append({
                "full_image_id": metadata["full_image_id"],
                "base_image_id": metadata["base_image_id"],
                "generated_caption": text,
                "original_caption": metadata["caption"],
                "expert_score": metadata["expert_score"],
                "crowd_score": metadata["crowd_score"],
                "dataset_index": valid_indices[idx]
            })
            
            # Comment translated to English and cleaned.
            evaluation_data["hypotheses"].append(text)
            evaluation_data["references"].append([metadata["caption"]])  # Comment translated to English and cleaned.
            
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
if not results_df.empty:
    print("\n5:")
    print(results_df.head())

# Comment translated to English and cleaned.
print("\n...")
evaluation_start = time.time()

# Comment translated to English and cleaned.
hypotheses = evaluation_data["hypotheses"]
references = evaluation_data["references"]  # Comment translated to English and cleaned.

# Comment translated to English and cleaned.
print("BLEU...")
smoothie = SmoothingFunction().method4

# Comment translated to English and cleaned.
bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothie)
bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
bleu3 = corpus_bleu(references, hypotheses, weights=(0.333, 0.333, 0.333, 0), smoothing_function=smoothie)
bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

# Comment translated to English and cleaned.
print("ROUGE...")
rouge = Rouge()
rouge_scores = rouge.get_scores(hypotheses, [ref[0] for ref in references], avg=True)

# Comment translated to English and cleaned.
print("METEOR...")
meteor_scores = []
for hyp, refs in zip(hypotheses, references):
    # Comment translated to English and cleaned.
    hyp_tokens = nltk.word_tokenize(hyp)
    ref_tokens = [nltk.word_tokenize(ref) for ref in refs]
    
    # Comment translated to English and cleaned.
    score = meteor_score(ref_tokens, hyp_tokens)
    meteor_scores.append(score)

meteor_avg = np.mean(meteor_scores)

# Comment translated to English and cleaned.
print("CIDEr...")
# Comment translated to English and cleaned.
# Comment translated to English and cleaned.
# Comment translated to English and cleaned.

def compute_cider(hypotheses, references, n=4):
    """
    CIDErn-gram
    """
    from collections import Counter
    cider_scores = []
    
    for hyp, refs in zip(hypotheses, references):
        # Comment translated to English and cleaned.
        hyp_tokens = nltk.word_tokenize(hyp.lower())
        ref_tokens_list = [nltk.word_tokenize(ref.lower()) for ref in refs]
        
        # Comment translated to English and cleaned.
        hyp_ngrams = []
        for i in range(1, n+1):
            hyp_ngrams.extend(' '.join(hyp_tokens[j:j+i]) for j in range(len(hyp_tokens)-i+1))
        
        ref_ngrams_list = []
        for ref_tokens in ref_tokens_list:
            ref_ngrams = []
            for i in range(1, n+1):
                ref_ngrams.extend(' '.join(ref_tokens[j:j+i]) for j in range(len(ref_tokens)-i+1))
            ref_ngrams_list.append(ref_ngrams)
        
        # Comment translated to English and cleaned.
        # Comment translated to English and cleaned.
        # Comment translated to English and cleaned.
        all_ngrams = set(hyp_ngrams)
        for ngrams in ref_ngrams_list:
            all_ngrams.update(ngrams)
        
        # Comment translated to English and cleaned.
        ngram_counts = Counter()
        for ngrams in ref_ngrams_list:
            ngram_counts.update(ngrams)
        
        # Comment translated to English and cleaned.
        weights = {ngram: 1.0 / np.log(2 + count) for ngram, count in ngram_counts.items()}
        
        # Comment translated to English and cleaned.
        hyp_vec = np.zeros(len(all_ngrams))
        ref_vecs = np.zeros((len(ref_ngrams_list), len(all_ngrams)))
        
        # Comment translated to English and cleaned.
        ngram_to_idx = {ngram: i for i, ngram in enumerate(all_ngrams)}
        
        # Comment translated to English and cleaned.
        for ngram in hyp_ngrams:
            if ngram in ngram_to_idx:
                idx = ngram_to_idx[ngram]
                hyp_vec[idx] += weights.get(ngram, 1.0)
        
        # Comment translated to English and cleaned.
        for i, ngrams in enumerate(ref_ngrams_list):
            for ngram in ngrams:
                if ngram in ngram_to_idx:
                    idx = ngram_to_idx[ngram]
                    ref_vecs[i, idx] += weights.get(ngram, 1.0)
        
        # Comment translated to English and cleaned.
        ref_avg = np.mean(ref_vecs, axis=0)
        
        # Comment translated to English and cleaned.
        dot_product = np.dot(hyp_vec, ref_avg)
        norm_hyp = np.linalg.norm(hyp_vec)
        norm_ref = np.linalg.norm(ref_avg)
        
        if norm_hyp > 0 and norm_ref > 0:
            cider_score = dot_product / (norm_hyp * norm_ref)
        else:
            cider_score = 0.0
            
        cider_scores.append(cider_score)
    
    return np.mean(cider_scores)

# Comment translated to English and cleaned.
cider_score = compute_cider(hypotheses, references)

# Comment translated to English and cleaned.
metrics = {
    "BLEU-1": bleu1,
    "BLEU-2": bleu2,
    "BLEU-3": bleu3,
    "BLEU-4": bleu4,
    "ROUGE": {
        "rouge-1": {
            "f": rouge_scores["rouge-1"]["f"],
            "p": rouge_scores["rouge-1"]["p"],
            "r": rouge_scores["rouge-1"]["r"]
        },
        "rouge-2": {
            "f": rouge_scores["rouge-2"]["f"],
            "p": rouge_scores["rouge-2"]["p"],
            "r": rouge_scores["rouge-2"]["r"]
        },
        "rouge-l": {
            "f": rouge_scores["rouge-l"]["f"],
            "p": rouge_scores["rouge-l"]["p"],
            "r": rouge_scores["rouge-l"]["r"]
        }
    },
    "METEOR": meteor_avg,
    "CIDEr": cider_score,
    "evaluation_time": time.time() - evaluation_start
}

# Comment translated to English and cleaned.
with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=4)

# Comment translated to English and cleaned.
print("\n:")
print(f"BLEU-1: {bleu1:.4f}")
print(f"BLEU-2: {bleu2:.4f}")
print(f"BLEU-3: {bleu3:.4f}")
print(f"BLEU-4: {bleu4:.4f}")
print(f"METEOR: {meteor_avg:.4f}")
print(f"CIDEr: {cider_score:.4f}")
print(f"ROUGE-1 F1: {rouge_scores['rouge-1']['f']:.4f}")
print(f"ROUGE-2 F1: {rouge_scores['rouge-2']['f']:.4f}")
print(f"ROUGE-L F1: {rouge_scores['rouge-l']['f']:.4f}")
print(f": {metrics['evaluation_time']:.2f} ")
print(f": {metrics_file}")
