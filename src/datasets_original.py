import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import re

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, caption_path, processor):
        """
        Args:
            image_dir: 图片目录路径
            caption_path: 标注文件路径或标注目录路径
            processor: BLIP-2的处理器
        """
        self.image_dir = image_dir
        self.processor = processor
        
        # 确定标注文件路径
        self.caption_file = self._find_caption_file(caption_path)
        
        # 读取描述文件
        self.captions = []
        with open(self.caption_file, 'r') as f:
            for line in f:
                # 跳过空行
                if not line.strip():
                    continue
                    
                # 处理不同格式的标注文件
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
                
                # 移除图片ID中的#符号和额外后缀
                img_id = self._clean_image_id(img_id)
                
                self.captions.append({"image": img_id, "caption": caption})
        
        # 转换为DataFrame
        self.captions = pd.DataFrame(self.captions)
        print(f"成功加载 {len(self.captions)} 个图片描述")
        print(f"第一张图片ID: {self.captions.iloc[0]['image']}")

    def _clean_image_id(self, img_id):
        """清理图片ID，移除多余的后缀"""
        # 移除图片ID中的#符号（如果存在）
        if '#' in img_id:
            img_id = img_id.split('#')[0]
        
        # 移除额外的数字后缀（如 .1, .2 等）
        if re.search(r'\.\d+\.jpg$', img_id):
            img_id = re.sub(r'\.\d+\.jpg$', '.jpg', img_id)
        
        # 确保图片文件名正确
        if not img_id.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_id += '.jpg'
        
        return img_id

    def _find_caption_file(self, caption_path):
        """查找标注文件"""
        # 如果是文件，直接返回
        if os.path.isfile(caption_path):
            return caption_path
        
        # 如果是目录，查找可能的标注文件
        possible_files = [
            "Flickr8k.token.txt",
            "Flickr8k.lemma.token.txt",
            "captions.txt",
            "annotations.txt"
        ]
        
        for file in possible_files:
            file_path = os.path.join(caption_path, file)
            if os.path.isfile(file_path):
                print(f"找到标注文件: {file_path}")
                return file_path
        
        # 列出目录内容帮助调试
        print(f"标注目录内容: {os.listdir(caption_path)}")
        raise FileNotFoundError(f"在目录 {caption_path} 中找不到标注文件")

    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        row = self.captions.iloc[idx]
        img_id = row["image"]
        caption = row["caption"]
        
        image_path = os.path.join(self.image_dir, img_id)
        
        # 处理可能的文件名变体
        if not os.path.exists(image_path):
            # 尝试可能的扩展名
            base_name = os.path.splitext(img_id)[0]
            possible_extensions = ['.jpg', '.JPG', '.jpeg', '.png', '.PNG']
            possible_paths = [image_path]
            
            for ext in possible_extensions:
                possible_paths.append(os.path.join(self.image_dir, base_name + ext))
            
            # 尝试小写文件名
            possible_paths.append(os.path.join(self.image_dir, img_id.lower()))
            
            # 尝试移除额外数字后缀的版本
            if re.search(r'\d+\.jpg$', base_name):
                clean_base = re.sub(r'\d+\.jpg$', '.jpg', base_name)
                for ext in possible_extensions:
                    possible_paths.append(os.path.join(self.image_dir, clean_base + ext))
            
            # 去重
            possible_paths = list(set(possible_paths))
            
            found = False
            for path in possible_paths:
                if os.path.exists(path):
                    image_path = path
                    found = True
                    print(f"使用替代路径: {path}")
                    break
            
            if not found:
                # 列出所有可能的路径供调试
                print(f"找不到图片: {img_id}")
                print(f"尝试了以下路径: {possible_paths}")
                
                # 检查目录中是否有类似文件
                similar_files = [f for f in os.listdir(self.image_dir) 
                                if f.startswith(img_id.split('.')[0])]
                print(f"目录中有 {len(similar_files)} 个类似文件: {similar_files[:5]}")
                
                # 跳过这个样本而不是终止整个进程
                print(f"跳过样本 {idx}: {img_id}")
                return self.__getitem__((idx + 1) % len(self))  # 尝试下一个样本
        
        image = Image.open(image_path).convert("RGB")
        
        inputs = self.processor(
            images=image,
            text="Question: What is happening in this image? Answer:",
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        return inputs

 