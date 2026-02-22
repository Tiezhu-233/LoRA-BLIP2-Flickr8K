import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import re

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, caption_path, processor):
        """
        Args:
            image_dir: 
            caption_path: 
            processor: BLIP-2
        """
        self.image_dir = image_dir
        self.processor = processor
        
        # Comment translated to English and cleaned.
        self.caption_file = self._find_caption_file(caption_path)
        
        # Comment translated to English and cleaned.
        self.captions = []
        with open(self.caption_file, 'r') as f:
            for line in f:
                # Comment translated to English and cleaned.
                if not line.strip():
                    continue
                    
                # Comment translated to English and cleaned.
                if '\t' in line:
                    img_id, caption = line.strip().split('\t', 1)
                elif ',' in line:
                    img_id, caption = line.strip().split(',', 1)
                else:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        img_id = parts[0]
                        caption = ' '.join(parts[1:])
                    else:
                        continue
                
                # Comment translated to English and cleaned.
                img_id = self._clean_image_id(img_id)
                
                self.captions.append({"image": img_id, "caption": caption})
        
        # Comment translated to English and cleaned.
        self.captions = pd.DataFrame(self.captions)
        print(f" {len(self.captions)} ")
        print(f"ID: {self.captions.iloc[0]['image']}")

    def _clean_image_id(self, img_id):
        """ID"""
        # Comment translated to English and cleaned.
        if '#' in img_id:
            img_id = img_id.split('#')[0]
        
        # Comment translated to English and cleaned.
        if re.search(r'\.\d+\.jpg$', img_id):
            img_id = re.sub(r'\.\d+\.jpg$', '.jpg', img_id)
        
        # Comment translated to English and cleaned.
        if not img_id.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_id += '.jpg'
        
        return img_id

    def _find_caption_file(self, caption_path):
        """"""
        # Comment translated to English and cleaned.
        if os.path.isfile(caption_path):
            return caption_path
        
        # Comment translated to English and cleaned.
        possible_files = [
            "Flickr8k.token.txt",
            "Flickr8k.lemma.token.txt",
            "captions.txt",
            "annotations.txt"
        ]
        
        for file in possible_files:
            file_path = os.path.join(caption_path, file)
            if os.path.isfile(file_path):
                print(f": {file_path}")
                return file_path
        
        # Comment translated to English and cleaned.
        print(f": {os.listdir(caption_path)}")
        raise FileNotFoundError(f" {caption_path} ")

    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        row = self.captions.iloc[idx]
        img_id = row["image"]
        caption = row["caption"]
        
        image_path = os.path.join(self.image_dir, img_id)
        
        # Comment translated to English and cleaned.
        if not os.path.exists(image_path):
            # Comment translated to English and cleaned.
            base_name = os.path.splitext(img_id)[0]
            possible_extensions = ['.jpg', '.JPG', '.jpeg', '.png', '.PNG']
            possible_paths = [image_path]
            
            for ext in possible_extensions:
                possible_paths.append(os.path.join(self.image_dir, base_name + ext))
            
            # Comment translated to English and cleaned.
            possible_paths.append(os.path.join(self.image_dir, img_id.lower()))
            
            # Comment translated to English and cleaned.
            if re.search(r'\d+\.jpg$', base_name):
                clean_base = re.sub(r'\d+\.jpg$', '.jpg', base_name)
                for ext in possible_extensions:
                    possible_paths.append(os.path.join(self.image_dir, clean_base + ext))
            
            # Comment translated to English and cleaned.
            possible_paths = list(set(possible_paths))
            
            found = False
            for path in possible_paths:
                if os.path.exists(path):
                    image_path = path
                    found = True
                    print(f": {path}")
                    break
            
            if not found:
                # Comment translated to English and cleaned.
                print(f": {img_id}")
                print(f": {possible_paths}")
                
                # Comment translated to English and cleaned.
                similar_files = [f for f in os.listdir(self.image_dir) 
                                if f.startswith(img_id.split('.')[0])]
                print(f" {len(similar_files)} : {similar_files[:5]}")
                
                # Comment translated to English and cleaned.
                print(f" {idx}: {img_id}")
                return self.__getitem__((idx + 1) % len(self))  # Comment translated to English and cleaned.
        
        image = Image.open(image_path).convert("RGB")
        
        inputs = self.processor(
            images=image,
            text="Question: What is happening in this image? Answer:",
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        return inputs

 
