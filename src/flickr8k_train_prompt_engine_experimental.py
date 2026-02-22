from flickr8k_dataset_full import Flickr8kDataset
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import os
import pandas as pd
from tqdm import tqdm
import time
import numpy as np
import nltk
import random
from collections import defaultdict
from PIL import Image
import json
from typing import List, Dict, Tuple
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.getenv("OUTPUT_DIR", os.path.join(os.getenv("PROJECT_ROOT", "."), "output")), "caption_generation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download necessary NLTK resources
logger.info("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Set base paths
base_dir = os.getenv("PROJECT_ROOT", ".")
image_dir = os.getenv("FLICKR8K_IMAGE_DIR", os.path.join(base_dir, "data", "flickr8k", "Flickr8k_Dataset"))
caption_path = os.getenv("FLICKR8K_CAPTION_FILE", os.path.join(base_dir, "data", "flickr8k", "captions.txt"))

# Initialize models and processors
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

try:
    logger.info("Loading BLIP-2 model...")
    processor = Blip2Processor.from_pretrained("blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "blip2-opt-2.7b",
        torch_dtype=torch.float16
    ).to(device)
    logger.info("BLIP-2 model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    exit(1)

# Load dataset
try:
    logger.info("Creating dataset instance...")
    dataset = Flickr8kDataset(
        image_dir=image_dir,
        caption_path=caption_path,
        processor=processor,
        split='all'
    )
    logger.info(f"Dataset created successfully with {len(dataset)} samples")
    
    # Test first sample
    logger.info("Testing first sample...")
    sample = dataset[0]
    logger.info(f"Input tensor shape: {sample['pixel_values'].shape}")
    
except Exception as e:
    logger.error(f"Error loading dataset: {str(e)}")
    exit(1)

# Create output directory
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "flickr8k_blip2_descriptions.csv")
prompt_analysis_file = os.path.join(output_dir, "prompt_analysis.json")
debug_log_file = os.path.join(output_dir, "debug_samples.json")

class EnhancedPromptEngine:
    def __init__(self, device, model, processor):
        self.device = device
        self.model = model
        self.processor = processor
        self.max_prompt_length = 120
        self.label_names = [
            'outdoor', 'indoor', 'nature', 'urban', 'person', 'animal',
            'food', 'vehicle', 'building', 'water', 'sky', 'plant', 
            'sports', 'electronic', 'furniture'
        ] 
        self.debug_samples = []

        # Comment translated to English and cleaned.
        self.label_classifier = self._init_classifier()
        
        
        
        # Enhanced prompt components
        self.base_prompts = {
            "descriptive": [
                "Describe this image in detail:",
                "Provide a comprehensive description of this scene:",
                "Enumerate the key visual elements in this picture:"
            ],
            "creative": [
                "Imagine a story behind this image:",
                "Create a fictional narrative based on this photo:",
                "Write a poetic interpretation of this scene:"
            ],
            "analytical": [
                "Analyze the composition of this image:",
                "Explain the visual elements and their relationships:",
                "Interpret the artistic techniques used in this photo:"
            ],
            "emotional": [
                "Describe the emotions evoked by this image:",
                "Capture the mood and atmosphere of this scene:",
                "Express the feelings this photograph conveys:"
            ],
            "concise": [
                "Summarize this image in one sentence:",
                "Provide a brief caption for this photo:",
                "Describe this picture succinctly:"
            ]
        }
        
        self.style_modifiers = {
            "descriptive": [
                "with precise technical details",
                "using objective, factual language",
                "with comprehensive coverage of all elements"
            ],
            "creative": [
                "in a whimsical, imaginative style",
                "using vivid metaphors and similes",
                "with a touch of magical realism"
            ],
            "analytical": [
                "from an art critic's perspective",
                "with attention to compositional techniques",
                "analyzing visual balance and harmony"
            ],
            "emotional": [
                "with deep emotional resonance",
                "emphasizing the human experience",
                "focusing on psychological impact"
            ],
            "concise": [
                "using minimal but impactful words",
                "with telegraphic precision",
                "in a headline-style format"
            ]
        }
        
        self.constraint_modifiers = [
            ("word_count", "using exactly {n} words", lambda: random.randint(5, 15)),
            ("color_requirement", "mentioning at least {n} colors", lambda: random.randint(2, 4)),
            ("adjective_requirement", "including {n} descriptive adjectives", lambda: random.randint(2, 5)),
            ("perspective", "from the perspective of a {role}", 
             lambda: random.choice(["child", "tourist", "artist", "scientist", "historian"])),
            ("temporal", "as if describing a {time} event", 
             lambda: random.choice(["future", "past", "historical", "contemporary"])),
            ("sensory", "engaging the {sense} senses", 
             lambda: random.choice(["visual", "auditory", "tactile", "olfactory", "all"]))
        ]
        
        self.semantic_guides = [
            "Pay special attention to lighting and shadows",
            "Consider the cultural context of the scene",
            "Analyze the spatial relationships between objects",
            "Note any human interactions or emotions",
            "Observe the background elements and their significance",
            "Identify the visual hierarchy and focal points",
            "Describe the textures and material qualities"
        ]
        
        self.diversity_requirements = [
            "Avoid common descriptive phrases",
            "Use unexpected vocabulary choices",
            "Include at least one unconventional observation",
            "Create original metaphors or similes",
            "Find unique perspectives others might miss",
            "Challenge conventional viewing habits"
        ]
        
        self.category_counts = {
            "descriptive": 0,
            "creative": 0,
            "analytical": 0,
            "emotional": 0,
            "concise": 0
        }
        
        self.voting_history = defaultdict(list)

    def _init_classifier(self):
        """"""
        # Comment translated to English and cleaned.
        vision_hidden_size = self.model.config.vision_config.hidden_size
        
        classifier = torch.nn.Sequential(
            torch.nn.Linear(vision_hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, len(self.label_names)),
            torch.nn.Sigmoid()  # Comment translated to English and cleaned.
        ).to(self.device)
        
        # Comment translated to English and cleaned.
        # classifier.load_state_dict(torch.load("classifier_weights.pth"))
        return classifier

    def extract_visual_features(self, image_path):
        try:
            logger.info(f"Feature extraction start: {os.path.basename(image_path)}")
            image = Image.open(image_path).convert("RGB")
            logger.debug(f"Image loaded and converted to RGB: size={image.size}")
        
        # Comment translated to English and cleaned.
            inputs = self.processor(images=image, return_tensors="pt")
            logger.debug(f"Processor inputs created: pixel_values.shape={inputs['pixel_values'].shape}")
        
        # Comment translated to English and cleaned.
            inputs = {k: v.to(self.device, torch.float16) for k, v in inputs.items()}
            logger.debug(f"Inputs moved to {self.device} with dtype float16")
        
            with torch.no_grad():
            # Comment translated to English and cleaned.
                vision_outputs = self.model.vision_model(**inputs)
                logger.debug(f"Vision model output: last_hidden_state.shape={vision_outputs.last_hidden_state.shape}")
            
            # Comment translated to English and cleaned.
                cls_features = vision_outputs.last_hidden_state[:, 0, :]
                logger.info(f"[CLS] token extracted: shape={cls_features.shape}")
        
        # Comment translated to English and cleaned.
            features = cls_features.squeeze(0)
            logger.debug(f"Features squeezed: final shape={features.shape}")
            return features
        
        except Exception as e:
            logger.error(f"Feature extraction failed for {os.path.basename(image_path)}: {str(e)}", exc_info=True)
            return None
        
    

    def predict_labels(self, visual_features, top_k=3):
    
        if visual_features is None:
            logger.warning("predict_labels:  None")
            return [], np.zeros(len(self.label_names))

    # Comment translated to English and cleaned.
        classifier_dtype = next(self.label_classifier.parameters()).dtype
        visual_features = visual_features.to(self.device, dtype=classifier_dtype)

        with torch.no_grad():
            logits = self.label_classifier(visual_features.unsqueeze(0))
            probs = logits[0].cpu().numpy()

    # Comment translated to English and cleaned.
        top_indices = np.argsort(probs)[::-1][:top_k]
        predicted = [self.label_names[i] for i in top_indices]

        logger.info(f"Top-{top_k} predicted labels: {predicted}")
        logger.debug(f"All probs: {probs.tolist()}")
        return predicted, probs



    def extract_keywords(self, image_path: str) -> List[str]:
        
        try:
            # Comment translated to English and cleaned.
            visual_features = self.extract_visual_features(image_path)
            
            # Comment translated to English and cleaned.
            labels, probs = self.predict_labels(visual_features)
            
            # Comment translated to English and cleaned.
            if not labels:
                # Comment translated to English and cleaned.
                return ["general", "scene", "photograph", "image", "picture"]
                
            # Comment translated to English and cleaned.
            sorted_indices = np.argsort(probs)[::-1]
            top_labels = [self.label_names[i] for i in sorted_indices[:1]]
            
            logger.debug(f"Extracted keywords for {os.path.basename(image_path)}: {top_labels}")
            return top_labels
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            return ["general", "scene", "image"]


    
    def select_category(self) -> str:
        """Select prompt category using dynamic balancing with logging"""
        min_count = min(self.category_counts.values())
        candidates = [k for k, v in self.category_counts.items() if v == min_count]
        selected_category = random.choice(candidates)
        self.category_counts[selected_category] += 1
        
        logger.debug(f"Selected prompt category: {selected_category} (counts: {self.category_counts})")
        return selected_category
    
    def build_constraint(self) -> str:
        """Build a dynamic constraint phrase with logging"""
        constraint_type, template, value_fn = random.choice(self.constraint_modifiers)
        
        if "{n}" in template:
            value = value_fn()
            constraint = template.format(n=value)
        elif "{role}" in template:
            value = value_fn()
            constraint = template.format(role=value)
        elif "{time}" in template:
            value = value_fn()
            constraint = template.format(time=value)
        elif "{sense}" in template:
            value = value_fn()
            constraint = template.format(sense=value)
        else:
            constraint = template
            
        logger.debug(f"Built constraint: {constraint} (type: {constraint_type})")
        return constraint
    
    def generate_prompt_variants(self, image_path: str, base_image_id: str, num_variants: int = 3) -> Tuple[List[str], List[str]]:
        """Generate multiple prompt variants for an image with detailed logging"""
        keywords = self.extract_keywords(image_path)
        keyword_str = ", ".join(keywords[:1])  # Limit to 5 keywords
        
        prompts = []
        categories = []
        
        logger.info(f"\nGenerating {num_variants} prompts for image: {os.path.basename(image_path)}")
        logger.info(f"Extracted keywords: {keywords}")
        
        for variant_num in range(num_variants):
            category = self.select_category()
            base_prompt = random.choice(self.base_prompts[category])
            style_mod = random.choice(self.style_modifiers[category])
            constraint = self.build_constraint()
            semantic_guide = random.choice(self.semantic_guides)
            diversity_req = random.choice(self.diversity_requirements)
            
            prompt_parts = [
                base_prompt,
                f"Keywords: {keyword_str}",
                semantic_guide,
                style_mod,
                diversity_req,
                constraint
            ]
            
            # Filter out empty parts and join
            prompt = ". ".join(p for p in prompt_parts if p) + "."
            
            # # Truncate if too long
            # if len(prompt) > self.max_prompt_length:
            #     original_prompt = prompt
            #     prompt = prompt[:self.max_prompt_length].rsplit(' ', 1)[0] + "."
            #     logger.warning(f"Prompt truncated from {len(original_prompt)} to {len(prompt)} characters")
            
            prompts.append(prompt)
            categories.append(category)
            
            logger.info(f"\nPrompt Variant {variant_num + 1} (Category: {category}):")
            logger.info(f"Base: {base_prompt}")
            logger.info(f"Keywords: {keyword_str}")
            logger.info(f"Semantic Guide: {semantic_guide}")
            logger.info(f"Style: {style_mod}")
            logger.info(f"Diversity: {diversity_req}")
            logger.info(f"Constraint: {constraint}")
            logger.info(f"Final Prompt: {prompt}")
            logger.info("-" * 50)
        
        # Store sample debug info for the first few images
        if len(self.debug_samples) < 10:
            self.debug_samples.append({
                "image_path": image_path,
                "base_image_id": base_image_id,
                "keywords": keywords,
                "prompts": prompts,
                "categories": categories
            })
        
        return prompts, categories
    
    def select_best_caption(self, captions: List[str], image_path: str) -> str:
        """Select best caption using simple heuristic with logging"""
        selected = max(captions, key=lambda x: len(x.split()))  # Prefer longer captions
        logger.info(f"\nCaption selection for {os.path.basename(image_path)}:")
        for i, cap in enumerate(captions):
            logger.info(f"Option {i+1}: {cap}")
        logger.info(f"Selected: {selected}")
        return selected
    
    def get_prompt_analysis(self) -> Dict:
        """Get detailed statistics about prompt usage"""
        return {
            "total_prompts_generated": sum(self.category_counts.values()),
            "category_distribution": self.category_counts.copy(),
            "most_used_category": max(self.category_counts, key=self.category_counts.get),
            "least_used_category": min(self.category_counts, key=self.category_counts.get),
            "constraints_used": [m[0] for m in self.constraint_modifiers],
            "style_modifiers_used": list(self.style_modifiers.keys())
        }

# Initialize enhanced prompt engine
prompt_engine = EnhancedPromptEngine(device, model, processor)

# Generation pipeline
logger.info(f"\nStarting description generation for entire dataset...")
model.eval()

results = []
batch_size = 4
num_batches = (len(dataset) + batch_size - 1) // batch_size

start_time = time.time()
skipped_indices = []
failed_batches = 0
max_length = 120

# Comment translated to English and cleaned.
first_caption_meta = None
first_caption_text = None


with torch.no_grad():
    progress_bar = tqdm(range(num_batches), desc="Generating descriptions",leave=True)
    for i in progress_bar:
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(dataset))
        
        batch_samples = []
        valid_indices = []
        metadata_list = []
        prompt_variants_list = []
        category_variants_list = []
        image_paths = []
        
        # Collect batch data
        for j in range(start_idx, end_idx):
            try:
                sample = dataset[j]
                metadata = dataset.get_metadata(j)
                base_id = metadata["base_image_id"]
                img_path = sample["image_path"]
                
                prompts, categories = prompt_engine.generate_prompt_variants(
                    img_path, base_id, num_variants=3
                )
                
                batch_samples.append(sample)
                valid_indices.append(j)
                metadata_list.append(metadata)
                prompt_variants_list.append(prompts)
                category_variants_list.append(categories)
                image_paths.append(img_path)
            except Exception as e:
                logger.error(f"Skipping sample {j}: {str(e)}", exc_info=True)
                skipped_indices.append(j)
        
        if not batch_samples:
            failed_batches += 1
            logger.warning(f"Batch {i} failed - no valid samples")
            continue
        
        # Process batch with consistent tensor sizes
        expanded_pixel_values = []
        expanded_input_ids = []
        expanded_attention_mask = []
        expanded_metadata = []
        
        for idx, sample in enumerate(batch_samples):
            metadata = metadata_list[idx]
            prompts = prompt_variants_list[idx]
            categories = category_variants_list[idx]
            
            for prompt, category in zip(prompts, categories):
                # Process text with fixed length
                text_inputs = processor(
                    text=prompt,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=max_length,
                    truncation=True
                )
                
                expanded_pixel_values.append(sample["pixel_values"])
                expanded_input_ids.append(text_inputs["input_ids"])
                expanded_attention_mask.append(text_inputs["attention_mask"])
                
                expanded_metadata.append({
                    "base_metadata": metadata,
                    "prompt": prompt,
                    "category": category,
                    "image_path": image_paths[idx]
                })

        
        
        # Verify tensor shapes before concatenation
        try:
            assert all(p.shape == expanded_pixel_values[0].shape for p in expanded_pixel_values)
            assert all(i.shape == expanded_input_ids[0].shape for i in expanded_input_ids)
            assert all(a.shape == expanded_attention_mask[0].shape for a in expanded_attention_mask)
            
            pixel_values = torch.cat(expanded_pixel_values).to(device)
            input_ids = torch.cat(expanded_input_ids).to(device)
            attention_mask = torch.cat(expanded_attention_mask).to(device)
        except (AssertionError, RuntimeError) as e:
            logger.error(f"Batch {i} failed shape validation: {str(e)}")
            logger.debug(f"Pixel shapes: {[p.shape for p in expanded_pixel_values]}")
            logger.debug(f"Input ID shapes: {[i.shape for i in expanded_input_ids]}")
            failed_batches += 1
            continue
        
        # Generate descriptions with diverse parameters
        try:
            logger.info(f"\nGenerating captions for batch {i} with {len(expanded_pixel_values)} prompts")
            
            generated_ids = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=75,
                # num_beams=3,
                early_stopping=True,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                no_repeat_ngram_size=3,
                temperature=0.7,
                length_penalty=1.2
            )
            
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            logger.info(f"Generated {len(generated_texts)} captions for batch {i}")
            
            for idx, (text, meta) in enumerate(zip(generated_texts, expanded_metadata)):
                clean_text = text.replace(meta["prompt"], "").strip()

            

            

                logger.debug(f"\nImage: {os.path.basename(meta['image_path'])}")
                logger.debug(f"Prompt: {meta['prompt']}")
                logger.debug(f"Generated: {clean_text}")
                
                results.append({
                    "image_id": meta["base_metadata"]["full_image_id"],
                    "base_image_id": meta["base_metadata"]["base_image_id"],
                    "generated_caption": clean_text,
                    "original_caption": meta["base_metadata"]["caption"],
                    "prompt_used": meta["prompt"],
                    "prompt_category": meta["category"],
                    "image_path": meta["image_path"],
                    "dataset_index": valid_indices[idx // 3]
                })
                # Comment translated to English and cleaned.
                if first_caption_meta is None and results:
                    first_caption_meta = results[0]  # Comment translated to English and cleaned.
                    first_caption_text = first_caption_meta["generated_caption"]

                    
                # Comment translated to English and cleaned.
                    logger.info("\n===  ===")
                    logger.info(f": {os.path.basename(first_caption_meta['image_path'])}")
                    logger.info(f" Prompt: {first_caption_meta['prompt_used']}")
                    logger.info(f" Caption: {first_caption_text}")
                    logger.info("=========================")
                

                
        except Exception as e:
            logger.error(f"Generation failed for batch {i}: {str(e)}", exc_info=True)
            failed_batches += 1
            continue
        
        progress_bar.set_postfix({
            "processed": f"{len(results)}/{len(dataset)*3}",
            "skipped": len(skipped_indices),
            "failed_batches": failed_batches
        })
# Comment translated to English and cleaned.
# first_caption_meta = None
# first_caption_text = None

# with torch.no_grad():
#     progress_bar = tqdm(range(num_batches), desc="Generating descriptions", leave=True)
#     for i in progress_bar:
#         start_idx = i * batch_size
#         end_idx = min((i + 1) * batch_size, len(dataset))
        
#         batch_samples = []
#         valid_indices = []
#         metadata_list = []
#         prompt_variants_list = []
#         category_variants_list = []
#         image_paths = []
        
#         # Collect batch data
#         for j in range(start_idx, end_idx):
#             try:
#                 sample = dataset[j]
#                 metadata = dataset.get_metadata(j)
#                 base_id = metadata["base_image_id"]
#                 img_path = sample["image_path"]
                
#                 prompts, categories = prompt_engine.generate_prompt_variants(
#                     img_path, base_id, num_variants=3
#                 )
                
#                 batch_samples.append(sample)
#                 valid_indices.append(j)
#                 metadata_list.append(metadata)
#                 prompt_variants_list.append(prompts)
#                 category_variants_list.append(categories)
#                 image_paths.append(img_path)
#             except Exception as e:
#                 logger.error(f"Skipping sample {j}: {str(e)}", exc_info=True)
#                 skipped_indices.append(j)
        
#         if not batch_samples:
#             failed_batches += 1
#             logger.warning(f"Batch {i} failed - no valid samples")
#             continue
        
# Comment translated to English and cleaned.
#         try:
# Comment translated to English and cleaned.
#             batch_images = []
#             batch_texts = []
#             expanded_metadata = []
            
#             for idx, sample in enumerate(batch_samples):
# Comment translated to English and cleaned.
#                 image = Image.open(sample["image_path"]).convert("RGB")
                
# Comment translated to English and cleaned.
#                 prompts = prompt_variants_list[idx]
#                 categories = category_variants_list[idx]
                
# Comment translated to English and cleaned.
#                 for prompt, category in zip(prompts, categories):
#                     batch_images.append(image)
#                     batch_texts.append(prompt)
                    
# Comment translated to English and cleaned.
#                     expanded_metadata.append({
#                         "base_metadata": metadata_list[idx],
#                         "prompt": prompt,
#                         "category": category,
#                         "image_path": image_paths[idx]
#                     })
            
#             logger.info(f"\nProcessing batch {i} with {len(batch_images)} image-prompt pairs")
            
#             encoding = processor(
#                 images=batch_images,
#                 text=batch_texts,
#                 return_tensors="pt",
#                 padding="max_length",
#                 truncation=True,
# Comment translated to English and cleaned.
#             )
#             pixel_values   = encoding["pixel_values"].to(device, torch.float16)
#             input_ids      = encoding["input_ids"].to(device)        # long
#             attention_mask = encoding["attention_mask"].to(device)   # long
# Comment translated to English and cleaned.
#             generated_ids = model.generate(
#                 pixel_values=pixel_values,
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 max_new_tokens=50,
#                 do_sample=True, top_k=50, top_p=0.95, temperature=0.7,
#                 no_repeat_ngram_size=2
#             )

            
# Comment translated to English and cleaned.
#             generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
#             logger.info(f"Generated {len(generated_texts)} captions for batch {i}")
            
# Comment translated to English and cleaned.
#             for idx, (text, meta) in enumerate(zip(generated_texts, expanded_metadata)):
# Comment translated to English and cleaned.
#                 clean_text = text.replace(meta["prompt"], "").strip()
                
#                 logger.debug(f"\nImage: {os.path.basename(meta['image_path'])}")
#                 logger.debug(f"Prompt: {meta['prompt']}")
#                 logger.debug(f"Generated: {clean_text}")
                
# Comment translated to English and cleaned.
#                 result_entry = {
#                     "image_id": meta["base_metadata"]["full_image_id"],
#                     "base_image_id": meta["base_metadata"]["base_image_id"],
#                     "generated_caption": clean_text,
#                     "original_caption": meta["base_metadata"]["caption"],
#                     "prompt_used": meta["prompt"],
#                     "prompt_category": meta["category"],
#                     "image_path": meta["image_path"],
#                     "dataset_index": valid_indices[idx // 3]
#                 }
                
                
# Comment translated to English and cleaned.
#                 results.append(result_entry)
                
# Comment translated to English and cleaned.
                
                    
#                 if first_caption_meta is None and results:
# Comment translated to English and cleaned.
#                     first_caption_text = first_caption_meta["generated_caption"]

                
# Comment translated to English and cleaned.
# Comment translated to English and cleaned.
# Comment translated to English and cleaned.
# Comment translated to English and cleaned.
#                     print("=========================")
                
#         except Exception as e:
#             logger.error(f"Generation failed for batch {i}: {str(e)}", exc_info=True)
#             failed_batches += 1
#             continue
        
#         progress_bar.set_postfix({
#             "processed": f"{len(results)}/{len(dataset)*3}",
#             "skipped": len(skipped_indices),
#             "failed_batches": failed_batches
#         })

# Save results with comprehensive data
results_df = pd.DataFrame(results)
if not results_df.empty:
    results_df.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to {output_file}")
    
    # Save sample results to debug log
    sample_results = results_df.head(10).to_dict('records')
    with open(debug_log_file, "w") as f:
        json.dump({
            "sample_results": sample_results,
            "debug_prompts": prompt_engine.debug_samples
        }, f, indent=4)
    logger.info(f"Debug samples saved to {debug_log_file}")
else:
    logger.warning("\nNo results to save!")

# Save detailed prompt analysis
full_analysis = {
    "execution_stats": {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "elapsed_seconds": time.time() - start_time,
        "total_samples": len(dataset) * 3,
        "processed_samples": len(results),
        "skipped_samples": len(skipped_indices),
        "failed_batches": failed_batches,
        "processing_rate": len(results) / (time.time() - start_time) if results else 0
    },
    "prompt_analysis": prompt_engine.get_prompt_analysis(),
    "system_info": {
        "device": device,
        "blip2_model": "blip2-opt-2.7b",
        "batch_size": batch_size,
        "max_prompt_length": max_length
    }
}

with open(prompt_analysis_file, "w") as f:
    json.dump(full_analysis, f, indent=4)

logger.info("\nFinal Statistics:")
logger.info(f"Total processing time: {full_analysis['execution_stats']['elapsed_seconds']:.2f} seconds")
logger.info(f"Successfully processed: {len(results)}/{len(dataset)*3} samples")
logger.info(f"Skipped samples: {len(skipped_indices)}")
logger.info(f"Failed batches: {failed_batches}")
logger.info("Prompt category distribution:")
for cat, count in prompt_engine.category_counts.items():
    logger.info(f"  {cat}: {count} prompts")
logger.info(f"\nFull analysis saved to {prompt_analysis_file}")

