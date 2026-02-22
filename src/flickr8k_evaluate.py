import json
import os
from collections import defaultdict

import nltk
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge


def _env(name, default):
    return os.getenv(name, default)


def ensure_nltk_data():
    nltk_data_dir = _env("NLTK_DATA", "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download("punkt_tab", quiet=True)


def load_pairs(test_json_path, pred_json_path):
    with open(test_json_path, "r", encoding="utf-8") as f:
        refs = json.load(f)
    with open(pred_json_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    refs_by_image = defaultdict(list)
    for item in refs:
        refs_by_image[item["image"]].append(str(item["caption"]).strip())

    hypotheses = []
    references = []
    for pred in preds:
        image = pred.get("image")
        hyp = str(pred.get("caption", "")).strip()
        if not image or not hyp or image not in refs_by_image:
            continue
        refs_for_image = [r for r in refs_by_image[image] if r]
        if not refs_for_image:
            continue
        hypotheses.append(hyp)
        references.append(refs_for_image)

    return hypotheses, references


def compute_cider(hypotheses, references, n=4):
    scores = []
    for hyp, ref_list in zip(hypotheses, references):
        hyp_tokens = nltk.word_tokenize(hyp.lower())
        ref_tokens_list = [nltk.word_tokenize(r.lower()) for r in ref_list]

        hyp_ngrams = []
        ref_ngrams = []

        for i in range(1, n + 1):
            hyp_ngrams.extend(" ".join(hyp_tokens[j : j + i]) for j in range(len(hyp_tokens) - i + 1))
            for ref_tokens in ref_tokens_list:
                ref_ngrams.extend(" ".join(ref_tokens[j : j + i]) for j in range(len(ref_tokens) - i + 1))

        hyp_counts = {ng: hyp_ngrams.count(ng) for ng in set(hyp_ngrams)}
        ref_counts = {ng: ref_ngrams.count(ng) for ng in set(ref_ngrams)}

        common = set(hyp_counts.keys()) & set(ref_counts.keys())
        overlap = sum(min(hyp_counts[ng], ref_counts[ng]) for ng in common)
        scores.append(overlap / max(len(hyp_ngrams), 1))

    return float(np.mean(scores)) if scores else 0.0


def main():
    ensure_nltk_data()

    test_json_path = _env("EVAL_TEST_JSON", _env("FLICKR8K_TEST_JSON", os.path.join("blip2_lora_finetune", "flickr8k_test.json")))
    pred_json_path = _env("EVAL_PRED_JSON", _env("INFER_OUTPUT_JSON", "generated_captions.json"))
    output_path = _env("EVAL_OUTPUT_JSON", os.path.join("output", "evaluation_metrics.json"))

    hypotheses, references = load_pairs(test_json_path, pred_json_path)
    if not hypotheses:
        raise RuntimeError("No valid prediction-reference pairs for evaluation.")

    tokenized_hypotheses = [nltk.word_tokenize(h) for h in hypotheses]
    tokenized_references = [[nltk.word_tokenize(r) for r in refs] for refs in references]

    smooth = SmoothingFunction().method4
    bleu1 = corpus_bleu(tokenized_references, tokenized_hypotheses, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(tokenized_references, tokenized_hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(tokenized_references, tokenized_hypotheses, weights=(1 / 3, 1 / 3, 1 / 3, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(tokenized_references, tokenized_hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    rouge = Rouge()
    rouge_scores = rouge.get_scores(hypotheses, [refs[0] for refs in references], avg=True)

    meteor_scores = [
        meteor_score([nltk.word_tokenize(r) for r in refs], nltk.word_tokenize(hyp))
        for hyp, refs in zip(hypotheses, references)
    ]
    meteor_avg = float(np.mean(meteor_scores))

    cider = compute_cider(hypotheses, references)

    metrics = {
        "num_samples": len(hypotheses),
        "BLEU-1": round(bleu1, 4),
        "BLEU-2": round(bleu2, 4),
        "BLEU-3": round(bleu3, 4),
        "BLEU-4": round(bleu4, 4),
        "ROUGE": rouge_scores,
        "METEOR": round(meteor_avg, 4),
        "CIDEr": round(cider, 4),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Evaluated {len(hypotheses)} samples")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
