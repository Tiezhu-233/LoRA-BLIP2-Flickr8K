import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import re
import torch
import numpy as np
from typing import Optional, Callable, Dict, Any, List, Union

class Flickr8kDataset(Dataset):
    def __init__(
        self, 
        image_dir: str, 
        caption_path: str, 
        processor: Callable,
        split: str = 'all', 
        use_lemma: bool = False,
        num_prompt_variants: int = 1,
        default_prompt: str = 'Identify which objects are present in the diagram and specify how they are related. Based on these findings, please generate a 15-30 word description for this image.',
        prompt_engine: Optional[Any] = None):
        
        self.image_dir = image_dir
        self.processor = processor
        self.split = split
        self.use_lemma = use_lemma
        self.prompt_engine = prompt_engine
        
        self.num_prompt_variants = max(1, num_prompt_variants)
        self.default_prompt = default_prompt
        self._prompt_cache = {}
        
        # Validate paths
        if not os.path.isdir(image_dir):
            raise ValueError(f"Image directory not found: {image_dir}")
        
        # Find and validate caption file
        self.caption_file = self._find_caption_file(caption_path)
        
        # Load data
        self.annotations = self._load_annotations()
        self.split_images = self._load_split_images()
        
        # Prepare data
        self.filtered_data = self._prepare_data()
        
        print(f"Loaded {len(self.filtered_data)} image-caption pairs")
        print(f"Split: {split}, Lemma: {use_lemma}")
        if self.prompt_engine:
            print(f"Using dynamic prompt engine with {num_prompt_variants} variants per image")

    def _find_caption_file(self, caption_path: str) -> str:
        """Find appropriate caption file with validation"""
        # Handle direct file path
        if os.path.isfile(caption_path):
            return caption_path
        
        # Determine target file based on lemma preference
        target_file = "Flickr8k.lemma.txt" if self.use_lemma else "Flickr8k.token.txt"
        search_files = [target_file, "captions.txt", "annotations.txt"]
        
        # Search in directory
        if os.path.isdir(caption_path):
            for file in search_files:
                file_path = os.path.join(caption_path, file)
                if os.path.isfile(file_path):
                    return file_path
            
            # Fallback to any flickr8k text file
            for file in os.listdir(caption_path):
                if "flickr8k" in file.lower() and file.lower().endswith(".txt"):
                    if (self.use_lemma and "lemma" in file.lower()) or \
                       (not self.use_lemma and "token" in file.lower()):
                        return os.path.join(caption_path, file)
        
        raise FileNotFoundError(f"No suitable caption file found in {caption_path}")

    def _load_annotations(self) -> List[Dict[str, str]]:
        """Load and parse caption annotations"""
        annotations = []
        with open(self.caption_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Handle different delimiters
                if '\t' in line:
                    parts = line.split('\t', 1)
                else:
                    parts = line.split(maxsplit=1)
                
                if len(parts) < 2:
                    continue
                    
                img_id, caption = parts
                annotations.append({
                    "full_image_id": img_id,
                    "caption": caption.strip()
                })
        return annotations

    def _load_split_images(self) -> Optional[set]:
        """Load dataset split definitions"""
        if self.split == 'all':
            return None
        
        split_files = {
            'train': 'Flickr_8k.trainImages.txt',
            'dev': 'Flickr_8k.devImages.txt',
            'test': 'Flickr_8k.testImages.txt'
        }
        
        base_dir = os.path.dirname(self.caption_file)
        split_file = os.path.join(base_dir, split_files.get(self.split, ''))
        
        if not os.path.exists(split_file):
            print(f"Split file not found: {split_file}. Using all data.")
            return None
        
        with open(split_file, 'r') as f:
            return {line.strip() for line in f if line.strip()}

    def _prepare_data(self) -> List[Dict[str, str]]:
        """Prepare data without any quality filtering"""
        prepared = []
        
        for ann in self.annotations:
            full_id = ann['full_image_id']
            base_id = self._get_base_image_id(full_id)
            
            # Apply split filter if specified
            if self.split_images and base_id not in self.split_images:
                continue
                    
            prepared.append({
                "full_image_id": full_id,
                "base_image_id": base_id,
                "caption": ann['caption']
            })
            
        return prepared

    def _get_base_image_id(self, img_id: str) -> str:
        """Extract clean base image ID"""
        # Remove caption index if present
        if '#' in img_id:
            base_id = img_id.split('#')[0]
        else:
            base_id = img_id
        
        # Ensure file extension
        if not base_id.lower().endswith(('.jpg', '.jpeg', '.png')):
            base_id += '.jpg'
            
        return base_id

    def _find_image_file(self, base_id: str) -> Optional[str]:
        """Robust image file finder with case insensitivity"""
        # Check direct match first
        direct_path = os.path.join(self.image_dir, base_id)
        if os.path.exists(direct_path):
            return direct_path
            
        # Check without extension
        base_name = os.path.splitext(base_id)[0]
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            test_path = os.path.join(self.image_dir, base_name + ext)
            if os.path.exists(test_path):
                return test_path
                
        # Final check with lowercase
        lower_path = os.path.join(self.image_dir, base_id.lower())
        if os.path.exists(lower_path):
            return lower_path
            
        return None

    def __len__(self) -> int:
        return len(self.filtered_data) * self.num_prompt_variants
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Calculate original data index and prompt variant index
        orig_idx = idx // self.num_prompt_variants
        variant_idx = idx % self.num_prompt_variants
        
        item = self.filtered_data[orig_idx]
        base_id = item["base_image_id"]
        caption = item["caption"]
        
        # Load image
        img_path = self._find_image_file(base_id)
        if not img_path:
            raise FileNotFoundError(f"Image not found: {base_id}")
            
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Error loading {img_path}: {str(e)}")
        
        # Determine prompt
        if self.prompt_engine:
            if orig_idx not in self._prompt_cache:
                try:
                    prompts, categories = self.prompt_engine.generate_prompt_variants(
                        img_path, 
                        base_id,
                        num_variants=self.num_prompt_variants
                    )
                    self._prompt_cache[orig_idx] = (prompts, categories)
                except Exception as e:
                    print(f"Error generating prompts for {base_id}: {str(e)}. Using default prompt.")
                    prompts = [self.default_prompt] * self.num_prompt_variants
                    categories = ["error_fallback"] * self.num_prompt_variants
                    self._prompt_cache[orig_idx] = (prompts, categories)
            
            prompts, categories = self._prompt_cache[orig_idx]
            text_input = prompts[variant_idx]
            prompt_category = categories[variant_idx]
        else:
            text_input = self.default_prompt
            prompt_category = "default"
        
        # Prepare BLIP-2 input
        inputs = self.processor(
            images=image,
            text=text_input,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Add metadata
        inputs["image_path"] = img_path
        inputs["caption"] = caption
        inputs["full_image_id"] = item["full_image_id"]
        inputs["prompt"] = text_input
        inputs["prompt_category"] = prompt_category
        inputs["variant_index"] = variant_idx
        inputs["original_index"] = orig_idx
        
        return inputs

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Retrieve comprehensive item metadata"""
        orig_idx = idx // self.num_prompt_variants
        variant_idx = idx % self.num_prompt_variants
        
        item = self.filtered_data[orig_idx]
        img_path = self._find_image_file(item["base_image_id"])
        
        metadata = {
            "full_image_id": item["full_image_id"],
            "base_image_id": item["base_image_id"],
            "caption": item["caption"],
            "image_path": img_path,
            "variant_index": variant_idx,
            "total_variants": self.num_prompt_variants,
            "citation": "Hodosh, M., Young, P., & Hockenmaier, J. (2013). "
                      "Framing Image Description as a Ranking Task: Data, "
                      "Models and Evaluation Metrics. JAIR, 47, 853-899."
        }
        
        if self.prompt_engine and img_path:
            try:
                if orig_idx in self._prompt_cache:
                    prompts, categories = self._prompt_cache[orig_idx]
                else:
                    prompts, categories = self.prompt_engine.generate_prompt_variants(
                        img_path, 
                        item["base_image_id"],
                        num_variants=self.num_prompt_variants
                    )
                metadata.update({
                    "prompt": prompts[variant_idx],
                    "prompt_category": categories[variant_idx],
                    "all_prompts": prompts,
                    "all_categories": categories
                })
            except Exception as e:
                metadata["prompt_error"] = str(e)
        
        return metadata

    def get_original_length(self) -> int:
        """Get number of original images (without prompt variants)"""
        return len(self.filtered_data)