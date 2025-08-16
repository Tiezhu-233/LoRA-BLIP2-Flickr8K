# evaluate.py

import pandas as pd
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import numpy as np
import os
import json


# 明确指定 NLTK 数据路径（必须与下载命令保持一致）
nltk_data_dir = "nltk_data"
nltk.data.path.append(nltk_data_dir)


# 下载 NLTK 所需资源
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ==== 配置路径 ====
base_dir = "/root/autodl-tmp"
csv_file = os.path.join(base_dir, "output", "flickr8k_blip2_descriptions.csv")
metrics_file = os.path.join(base_dir, "output", "evaluation_metrics.json")

# ==== 加载生成结果 ====
df = pd.read_csv(csv_file)
hypotheses = df["generated_caption"].fillna("").astype(str).tolist()
references = df["original_caption"].fillna("").astype(str).tolist()

# ==== 过滤空生成 ====
filtered_pairs = [
    (h, [r]) for h, r in zip(hypotheses, references) if h.strip()
]

if not filtered_pairs:
    print("所有生成的 caption 为空，无法评估。")
    exit()

# 拆分为两个列表
hypotheses, references = zip(*filtered_pairs)
hypotheses = list(hypotheses)
references = list(references)  # references 为 List[List[str]]

# ==== BLEU 分数 ====
print("计算 BLEU...")
smoothie = SmoothingFunction().method4
bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothie)
bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
bleu3 = corpus_bleu(references, hypotheses, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothie)
bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

# ==== ROUGE 分数 ====
print("计算 ROUGE...")
rouge = Rouge()
ref_texts = [r[0] for r in references]
rouge_scores = rouge.get_scores(hypotheses, ref_texts, avg=True)

# ==== METEOR 分数 ====
print("计算 METEOR...")
meteor_scores = [
    meteor_score([nltk.word_tokenize(ref)], nltk.word_tokenize(hyp))
    for hyp, [ref] in zip(hypotheses, references)
]


# ==== CIDEr（简化版）====
print("计算 CIDEr（简化）...")
def compute_cider(hyps, refs, n=4):
    scores = []
    for hyp, ref_list in zip(hyps, refs):
        hyp_tokens = nltk.word_tokenize(hyp.lower())
        ref_tokens = [nltk.word_tokenize(r.lower()) for r in ref_list]

        hyp_ngrams = []
        ref_ngrams = []

        for i in range(1, n+1):
            hyp_ngrams += [' '.join(hyp_tokens[j:j+i]) for j in range(len(hyp_tokens)-i+1)]
            for ref in ref_tokens:
                ref_ngrams += [' '.join(ref[j:j+i]) for j in range(len(ref)-i+1)]

        hyp_counts = {ng: hyp_ngrams.count(ng) for ng in set(hyp_ngrams)}
        ref_counts = {ng: ref_ngrams.count(ng) for ng in set(ref_ngrams)}

        common = set(hyp_counts.keys()) & set(ref_counts.keys())
        score = sum(min(hyp_counts[ng], ref_counts[ng]) for ng in common)

        norm = max(len(hyp_ngrams), 1)
        scores.append(score / norm)

    return float(np.mean(scores))

cider_score = compute_cider(hypotheses, references)

# ==== 汇总评估指标 ====
metrics = {
    "BLEU-1": round(bleu1, 4),
    "BLEU-2": round(bleu2, 4),
    "BLEU-3": round(bleu3, 4),
    "BLEU-4": round(bleu4, 4),
    "ROUGE": {
        "rouge-1": rouge_scores["rouge-1"],
        "rouge-2": rouge_scores["rouge-2"],
        "rouge-l": rouge_scores["rouge-l"]
    },
    "METEOR": round(meteor_avg, 4),
    "CIDEr": round(cider_score, 4)
}

# ==== 保存到 JSON ====
with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=4)

# ==== 打印结果 ====
print("\n评估完成，结果如下:")
for k, v in metrics.items():
    if isinstance(v, dict):
        print(f"{k}:")
        for subk, subv in v.items():
            print(f"  {subk}: F1={subv['f']:.4f}, P={subv['p']:.4f}, R={subv['r']:.4f}")
    else:
        print(f"{k}: {v:.4f}")

print(f"\n指标已保存至: {metrics_file}")
