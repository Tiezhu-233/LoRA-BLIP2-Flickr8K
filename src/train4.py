# 定义提示词模板合集
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

from datasets2 import Flickr8kDataset
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

# 下载必要的NLTK资源
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# 设置基础路径
base_dir = "/root/autodl-tmp"

# 设置数据集路径
image_dir = os.path.join(base_dir, "data/Flickr8k/Images")
caption_path = os.path.join(base_dir, "data/Flickr8k/captions.txt")

# 初始化模型和处理器
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

try:
    print("正在加载模型...")
    processor = Blip2Processor.from_pretrained("blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "blip2-opt-2.7b",
        torch_dtype=torch.float16
    ).to(device)
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    exit(1)

# 加载数据集
try:
    print("正在创建数据集实例...")
    dataset = Flickr8kDataset(
        image_dir=image_dir,
        caption_path=caption_path,
        processor=processor
    )
    print(f"数据集创建成功，包含 {len(dataset)} 个样本")
    
    # 测试第一个样本
    print("测试第一个样本...")
    sample = dataset[0]
    print("样本键:", list(sample.keys()))
    
except Exception as e:
    print(f"加载数据集时出错: {str(e)}")
    exit(1)

# 创建输出目录
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "flickr8k_blip2_descriptions.csv")
metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
prompt_stats_file = os.path.join(output_dir, "prompt_usage_stats.json")

# 生成描述
print(f"\n开始为整个数据集生成描述...")
print(f"使用 {len(PROMPT_TEMPLATES)} 种提示词模板")
model.eval()

results = []
batch_size = 8
num_batches = (len(dataset) + batch_size - 1) // batch_size

start_time = time.time()
skipped_indices = []  # 记录跳过的样本索引

# 记录提示词使用情况
prompt_usage = {prompt: 0 for prompt in PROMPT_TEMPLATES}

with torch.no_grad():
    progress_bar = tqdm(range(num_batches), desc="生成描述")
    for i in progress_bar:
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(dataset))
        
        # 获取批次数据
        batch_images = []
        valid_indices = []
        captions = []  # 存储参考描述
        image_ids = []  # 存储图片ID
        prompt_list = []  # 存储每个样本使用的提示词
        
        for j in range(start_idx, end_idx):
            try:
                # 获取样本数据
                sample = dataset[j]
                batch_images.append(sample["image"])
                valid_indices.append(j)
                image_ids.append(sample["image_id"])
                
                # 从数据集获取参考描述
                row = dataset.captions.iloc[j]
                captions.append(row["caption"])
                
                # 随机选择一个提示词模板
                selected_prompt = random.choice(PROMPT_TEMPLATES)
                prompt_list.append(selected_prompt)
                prompt_usage[selected_prompt] += 1
                
            except Exception as e:
                print(f"\n跳过样本 {j}: {str(e)}")
                skipped_indices.append(j)
        
        if not batch_images:
            continue
        
        # 使用处理器处理图片和文本
        inputs = processor(
            images=batch_images,
            text=prompt_list,
            padding="max_length",
            max_length=40,  # 与之前一致
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # 生成描述
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            num_beams=3,
            early_stopping=True
        )
        
        # 解码描述
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # 保存结果
        for idx, (text, caption, img_id, prompt) in enumerate(zip(generated_texts, captions, image_ids, prompt_list)):
            # 移除提示词（如果存在）
            if prompt in text:
                text = text.replace(prompt, "").strip()
            
            # 添加到结果列表
            results.append({
                "image_id": img_id,
                "generated_caption": text,
                "original_caption": caption,
                "prompt_used": prompt,
                "dataset_index": valid_indices[idx]
            })
            
        # 更新进度条
        progress_bar.set_postfix({
            "已处理": f"{len(results)}/{len(dataset)}",
            "跳过": len(skipped_indices)
        })

# 保存结果到CSV
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)

# 计算统计信息
elapsed_time = time.time() - start_time
processed_count = len(results)
if processed_count > 0:
    images_per_sec = processed_count / elapsed_time
else:
    images_per_sec = 0

print("\n生成完成!")
print(f"总样本数: {len(dataset)}")
print(f"成功处理: {processed_count}")
print(f"跳过样本: {len(skipped_indices)}")
print(f"总耗时: {elapsed_time:.2f} 秒")
print(f"处理速度: {images_per_sec:.2f} 图片/秒")
print(f"结果保存至: {output_file}")

# 保存跳过的样本信息
if skipped_indices:
    skipped_file = os.path.join(output_dir, "skipped_indices.txt")
    with open(skipped_file, 'w') as f:
        for idx in skipped_indices:
            f.write(f"{idx}\n")
    print(f"跳过的样本索引保存至: {skipped_file}")

# 保存提示词使用统计
with open(prompt_stats_file, 'w') as f:
    json.dump(prompt_usage, f, indent=2)
print(f"提示词使用统计保存至: {prompt_stats_file}")

# 打印提示词使用情况
print("\n提示词使用统计:")
for prompt, count in sorted(prompt_usage.items(), key=lambda x: x[1], reverse=True):
    print(f"{count}次: {prompt}")

# 打印前5个结果
if not results_df.empty:
    print("\n前5个生成描述:")
    print(results_df[["image_id", "prompt_used", "generated_caption"]].head())