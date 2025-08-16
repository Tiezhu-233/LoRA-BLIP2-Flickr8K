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
output_file = os.path.join(output_dir, "flickr8k_blip2_descriptions.csv")
metrics_file = os.path.join(output_dir, "evaluation_metrics.json")

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

# 存储用于评估的数据
evaluation_data = {
    "hypotheses": [],  # 生成的描述
    "references": []   # 参考描述（每个样本对应一个参考描述）
}

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
        
        # 生成描述
        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=50,
            num_beams=3,
            early_stopping=True
        )
        
        # 解码描述
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # 保存结果
        for idx, (text, metadata) in enumerate(zip(generated_texts, metadata_list)):
            # 移除提示词（如果存在）
            if "Question:" in text and "Answer:" in text:
                text = text.split("Answer:")[1].strip()
            
            # 添加到结果列表
            results.append({
                "full_image_id": metadata["full_image_id"],
                "base_image_id": metadata["base_image_id"],
                "generated_caption": text,
                "original_caption": metadata["caption"],
                "expert_score": metadata["expert_score"],
                "crowd_score": metadata["crowd_score"],
                "dataset_index": valid_indices[idx]
            })
            
            # 添加到评估数据
            evaluation_data["hypotheses"].append(text)
            evaluation_data["references"].append([metadata["caption"]])  # 单个参考描述
            
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

# 打印前5个结果
if not results_df.empty:
    print("\n前5个生成描述:")
    print(results_df.head())

# 计算评估指标
print("\n开始计算评估指标...")
evaluation_start = time.time()

# 1. 准备评估数据
hypotheses = evaluation_data["hypotheses"]
references = evaluation_data["references"]  # 每个假设对应一个参考描述列表（包含单个参考描述）

# 2. 计算BLEU分数
print("计算BLEU分数...")
smoothie = SmoothingFunction().method4

# 计算各种BLEU变体
bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothie)
bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
bleu3 = corpus_bleu(references, hypotheses, weights=(0.333, 0.333, 0.333, 0), smoothing_function=smoothie)
bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

# 3. 计算ROUGE分数
print("计算ROUGE分数...")
rouge = Rouge()
rouge_scores = rouge.get_scores(hypotheses, [ref[0] for ref in references], avg=True)

# 4. 计算METEOR分数
print("计算METEOR分数...")
meteor_scores = []
for hyp, refs in zip(hypotheses, references):
    # 将句子分词
    hyp_tokens = nltk.word_tokenize(hyp)
    ref_tokens = [nltk.word_tokenize(ref) for ref in refs]
    
    # 计算METEOR
    score = meteor_score(ref_tokens, hyp_tokens)
    meteor_scores.append(score)

meteor_avg = np.mean(meteor_scores)

# 5. 计算CIDEr分数
print("计算CIDEr分数...")
# 注意：完整的CIDEr实现需要TF或专门库，这里使用简化版
# 实际项目中建议使用pycocoevalcap
# 这里我们使用基于n-gram的简化版本

def compute_cider(hypotheses, references, n=4):
    """
    简化的CIDEr计算（基于n-gram共现）
    """
    from collections import Counter
    cider_scores = []
    
    for hyp, refs in zip(hypotheses, references):
        # 分词
        hyp_tokens = nltk.word_tokenize(hyp.lower())
        ref_tokens_list = [nltk.word_tokenize(ref.lower()) for ref in refs]
        
        # 计算n-gram
        hyp_ngrams = []
        for i in range(1, n+1):
            hyp_ngrams.extend(' '.join(hyp_tokens[j:j+i]) for j in range(len(hyp_tokens)-i+1))
        
        ref_ngrams_list = []
        for ref_tokens in ref_tokens_list:
            ref_ngrams = []
            for i in range(1, n+1):
                ref_ngrams.extend(' '.join(ref_tokens[j:j+i]) for j in range(len(ref_tokens)-i+1))
            ref_ngrams_list.append(ref_ngrams)
        
        # 计算TF-IDF权重（简化版）
        # 在实际CIDEr中，这是基于整个语料库计算的
        # 这里我们使用简化版本：1/log(2 + count)
        all_ngrams = set(hyp_ngrams)
        for ngrams in ref_ngrams_list:
            all_ngrams.update(ngrams)
        
        # 计算ngram在参考描述中出现的次数
        ngram_counts = Counter()
        for ngrams in ref_ngrams_list:
            ngram_counts.update(ngrams)
        
        # 计算权重
        weights = {ngram: 1.0 / np.log(2 + count) for ngram, count in ngram_counts.items()}
        
        # 计算向量
        hyp_vec = np.zeros(len(all_ngrams))
        ref_vecs = np.zeros((len(ref_ngrams_list), len(all_ngrams)))
        
        # 创建ngram到索引的映射
        ngram_to_idx = {ngram: i for i, ngram in enumerate(all_ngrams)}
        
        # 填充假设向量
        for ngram in hyp_ngrams:
            if ngram in ngram_to_idx:
                idx = ngram_to_idx[ngram]
                hyp_vec[idx] += weights.get(ngram, 1.0)
        
        # 填充参考向量
        for i, ngrams in enumerate(ref_ngrams_list):
            for ngram in ngrams:
                if ngram in ngram_to_idx:
                    idx = ngram_to_idx[ngram]
                    ref_vecs[i, idx] += weights.get(ngram, 1.0)
        
        # 计算参考向量的平均值
        ref_avg = np.mean(ref_vecs, axis=0)
        
        # 计算余弦相似度
        dot_product = np.dot(hyp_vec, ref_avg)
        norm_hyp = np.linalg.norm(hyp_vec)
        norm_ref = np.linalg.norm(ref_avg)
        
        if norm_hyp > 0 and norm_ref > 0:
            cider_score = dot_product / (norm_hyp * norm_ref)
        else:
            cider_score = 0.0
            
        cider_scores.append(cider_score)
    
    return np.mean(cider_scores)

# 计算CIDEr
cider_score = compute_cider(hypotheses, references)

# 6. 收集所有指标
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

# 7. 保存指标
with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=4)

# 8. 打印结果
print("\n评估结果:")
print(f"BLEU-1: {bleu1:.4f}")
print(f"BLEU-2: {bleu2:.4f}")
print(f"BLEU-3: {bleu3:.4f}")
print(f"BLEU-4: {bleu4:.4f}")
print(f"METEOR: {meteor_avg:.4f}")
print(f"CIDEr: {cider_score:.4f}")
print(f"ROUGE-1 F1: {rouge_scores['rouge-1']['f']:.4f}")
print(f"ROUGE-2 F1: {rouge_scores['rouge-2']['f']:.4f}")
print(f"ROUGE-L F1: {rouge_scores['rouge-l']['f']:.4f}")
print(f"评估耗时: {metrics['evaluation_time']:.2f} 秒")
print(f"指标保存至: {metrics_file}")