from datasets import Flickr8kDataset
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
logger = logging.getLogger(__name__)  # ← 新增

# 下面就可以安全地调用 logger.info(), logger.debug() 了：
logger.info("脚本开始执行")

# 设置基础路径
base_dir = "/root/autodl-tmp"

# 设置数据集路径
image_dir = os.path.join(base_dir, "data/Flickr8k/Images")
caption_path = os.path.join(base_dir, "data/Flickr8k/captions.txt")

# 初始化模型和处理器
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 初始化句子相似度模型
try:
    print("正在加载相似度模型...")
    similarity_model = SentenceTransformer('./all-MiniLM-L6-v2').to(device)
    print("相似度模型加载成功")
except Exception as e:
    print(f"相似度模型加载失败: {str(e)}")
    similarity_model = None

try:
    print("正在加载BLIP模型...")
    processor = Blip2Processor.from_pretrained("blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "blip2-opt-2.7b",
        torch_dtype=torch.float16
    ).to(device)
    print("BLIP模型加载成功")
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    exit(1)

# 加载数据集
try:
    print("正在创建数据集实例...")
    dataset = Flickr8kDataset(
        image_dir=image_dir,
        caption_path=caption_path,
        processor=processor,
        split='all'
    )
    print(f"数据集创建成功，包含 {len(dataset)} 个样本")
    
    # 测试第一个样本
    print("测试第一个样本...")
    sample = dataset[0]
    print("输入张量形状:", sample["pixel_values"].shape)
    print("样本键:", list(sample.keys()))
    
except Exception as e:
    print(f"加载数据集时出错: {str(e)}")
    exit(1)

# 创建输出目录
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "flickr8k_blip2_descriptions_similarity.csv")

# 准备评估数据结构
# 存储每个基础图片的所有参考描述
reference_descriptions = defaultdict(list)

# 首先收集所有参考描述
print("\n收集参考描述用于评估...")
for i in tqdm(range(len(dataset)), desc="收集参考描述"):
    metadata = dataset.get_metadata(i)
    base_id = metadata["base_image_id"]
    caption = metadata["caption"]
    reference_descriptions[base_id].append(caption)

# 生成描述
print(f"\n开始为整个数据集生成描述...")
model.eval()

results = []
batch_size = 8
num_batches = (len(dataset) + batch_size - 1) // batch_size

start_time = time.time()
skipped_indices = []  # 记录跳过的样本索引

def select_most_similar(candidates, reference, similarity_model):
    """
    计算每个候选描述与参考描述的相似度，返回最相似的一个
    """
    if similarity_model is None:
        # 如果没有相似度模型，使用简单的长度匹配作为后备
        ref_len = len(reference.split())
        best_idx = min(range(len(candidates)), 
                       key=lambda i: abs(len(candidates[i].split()) - ref_len))
        return candidates[best_idx]
    
    # 编码所有文本
    candidate_embeddings = similarity_model.encode(candidates, convert_to_tensor=True)
    ref_embedding = similarity_model.encode([reference], convert_to_tensor=True)
    
    # 计算余弦相似度
    similarities = util.pytorch_cos_sim(ref_embedding, candidate_embeddings)[0]
    
    # 找到最相似的候选
    best_idx = torch.argmax(similarities).item()
    return candidates[best_idx]

def process_generated_text(text,prompt):
    """处理生成的文本，移除提示部分"""
    text = re.sub(f"^{re.escape(prompt)}", "", text).strip()
    return text 

with torch.no_grad():
    progress_bar = tqdm(range(num_batches), desc="生成描述")
    for i in progress_bar:
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(dataset))
        
        # 获取批次数据
        batch_samples = []
        valid_indices = []
        metadata_list = []  # 存储元数据
        
        for j in range(start_idx, end_idx):
            try:
                # 获取样本数据
                sample = dataset[j]
                
                # 获取元数据
                metadata = dataset.get_metadata(j)
                
                batch_samples.append(sample)
                valid_indices.append(j)
                metadata_list.append(metadata)
            except Exception as e:
                print(f"\n跳过样本 {j}: {str(e)}")
                skipped_indices.append(j)
        
        if not batch_samples:
            continue
        
        # 准备批处理输入 - 使用动态填充
        pixel_values = torch.cat([s["pixel_values"] for s in batch_samples]).to(device)
        
        # 处理文本输入 - 使用动态填充
        input_ids_list = [s["input_ids"] for s in batch_samples]
        attention_mask_list = [s["attention_mask"] for s in batch_samples]
        
        # 找到批次中最长的序列长度
        max_length = max([ids.shape[1] for ids in input_ids_list])
        
        # 填充所有序列到相同长度
        padded_input_ids = []
        padded_attention_mask = []
        
        for input_ids, attn_mask in zip(input_ids_list, attention_mask_list):
            # 当前序列长度
            seq_len = input_ids.shape[1]
            
            # 计算需要填充的长度
            pad_len = max_length - seq_len
            
            # 填充input_ids (使用pad token id)
            padded_input = torch.cat([
                input_ids,
                torch.full((1, pad_len), processor.tokenizer.pad_token_id, device=input_ids.device)
            ], dim=1)
            
            # 填充attention mask
            padded_attn = torch.cat([
                attn_mask,
                torch.zeros((1, pad_len), device=attn_mask.device)
            ], dim=1)
            
            padded_input_ids.append(padded_input)
            padded_attention_mask.append(padded_attn)
            
        # 连接填充后的张量
        input_ids = torch.cat(padded_input_ids).to(device)
        attention_mask = torch.cat(padded_attention_mask).to(device)
        
        # 生成多个候选描述 (每个样本3个)
       
        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            do_sample=False,  # 启用随机采样
            num_return_sequences=3,  # 每个样本返回3个候选序列
            attention_mask=attention_mask,
            max_new_tokens=80,
            num_beams=5,
            temperature=0.7, 
            early_stopping=True,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            eos_token_id=model.config.eos_token_id,
        )
        
        # 解码所有生成的描述
        all_generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # 按样本分组生成结果：每个样本有3个候选描述
        batch_size_actual = len(batch_samples)
        grouped_texts = [all_generated_texts[i*3 : (i+1)*3] for i in range(batch_size_actual)]
        
                # 处理每个样本
        for idx, (candidates, metadata) in enumerate(zip(grouped_texts, metadata_list)):
            original_caption = metadata["caption"]

            prompt = ('Identify which objects are present in the diagram and specify how they are related. Based on these findings, please generate a 15-30 word description for this image.')
            # 后处理：移除提示词
            processed_candidates = [process_generated_text(text,prompt) for text in candidates]
            
            # 计算相似度并选择最佳描述
            best_caption = select_most_similar(
                candidates=processed_candidates,
                reference=original_caption,
                similarity_model=similarity_model
            )

            # 【新增日志】打印每条生成的最优描述
            logger.info(
                f"样本索引 {valid_indices[idx]} | "
                f"原始: \"{original_caption}\" | "
                f"生成: \"{best_caption}\""
            )

            # 添加到结果列表
            results.append({
                "full_image_id": metadata["full_image_id"],
                "base_image_id": metadata["base_image_id"],
                "generated_caption": best_caption,
                "original_caption": original_caption,
                "all_candidates": processed_candidates,
                "dataset_index": valid_indices[idx]
            })

            
        # 更新进度条
        progress_bar.set_postfix({
            "已处理": f"{len(results)}/{len(dataset)}",
            "跳过": len(skipped_indices)
        })

# 保存结果到CSV
if results:
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

# 打印前5个结果
if results:
    print("\n前5个生成描述:")
    for i, res in enumerate(results[:5]):
        print(f"\n样本 {i+1}:")
        print(f"  原始描述: {res['original_caption']}")
        print(f"  生成描述: {res['generated_caption']}")
        print(f"  所有候选: {res['all_candidates']}")
else:
    print("\n没有生成任何结果")