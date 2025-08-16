import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import random

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, caption_path, processor, max_length=40):
        """
        Flickr8k 数据集加载器
        
        参数:
            image_dir (str): 包含图片的目录路径
            caption_path (str): 包含图片描述的CSV文件路径
            processor (Blip2Processor): BLIP-2 处理器
            max_length (int): 文本的最大长度
        """
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        self.caption_file = self._find_caption_file(caption_path)
        # 加载描述文件
        self.captions = pd.read_csv(caption_file)
        
        # 确保所有图片都存在
        self.valid_indices = []
        for idx, row in self.captions.iterrows():
            img_path = os.path.join(self.image_dir, row['image'])
            if os.path.exists(img_path):
                self.valid_indices.append(idx)
        
        print(f"总样本数: {len(self.captions)}, 有效样本数: {len(self.valid_indices)}")
    
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
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # 获取实际索引
        real_idx = self.valid_indices[idx]
        row = self.captions.iloc[real_idx]
        
        # 加载图片
        img_path = os.path.join(self.image_dir, row['image'])
        image = Image.open(img_path).convert('RGB')
        
        # 返回原始图片和索引
        return {
            "image": image,
            "dataset_index": real_idx,
            "image_id": row['image']
        }



