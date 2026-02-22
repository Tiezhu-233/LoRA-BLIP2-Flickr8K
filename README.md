# BLIP-2 LoRA Fine-Tuning on Flickr8K

## Abstract
Large vision-language models often struggle with fine-grained image-text alignment in low-resource settings, leading to mode collapse and reduced output diversity. We address this by applying LoRA-based fine-tuning to BLIP-2 on Flickr8K, fine-tuning less than 5% of parameters. On the 2,000-image evaluation set, the lightweight approach consistently improves BLEU, ROUGE, and CIDEr scores while mitigating mode collapse, outperforming prompt-engineering and sampling baselines. Results demonstrate LoRA's effectiveness for robust, fine-grained alignment under data constraints.

## Model Architecture & Method
BLIP-2 (Bootstrapping Language-Image Pre-training) is an efficient vision-language framework that bridges a frozen vision encoder and a frozen large language model through a lightweight Querying Transformer (Q-Former):

- Vision Encoder: A pre-trained visual backbone (for example, ViT-based) extracts high-dimensional image features.
- Q-Former: Uses learnable query vectors to transform visual embeddings into language-compatible representations and enables cross-modal interaction.
- Language Model: A frozen pre-trained language model (for example, OPT/Flan-T5 style backbones) generates natural language descriptions.

To improve task-specific performance, this project applies LoRA (Low-Rank Adaptation):

- Method: Freeze most pre-trained parameters and insert low-rank trainable adapters into selected projection layers of the Q-Former and language model.
- Training data: 6,000 task-specific Flickr8K training images.
- Outcome: Better alignment quality and more stable training under limited data.

## Results & Analysis
Evaluation compares the zero-shot baseline against the LoRA fine-tuned model with BLEU, ROUGE, METEOR, and CIDEr metrics.

Main observations:
- Significant improvements across multiple captioning metrics.
- Stable optimization with a smoother loss curve.
- Reduced mode collapse behavior.
- Better grammatical quality and visual-text alignment.
- Slight vocabulary-diversity trade-off (lower TTR) but better relevance and accuracy.

![Qualitative Example](docs/qualitative_example.jpg)

Qualitative example:
- Without fine-tuning: `a person sitting on the ground with a tv in front of them`
- With fine-tuning: `A man is sitting in front of a television with a plate of food in front of him.`


## Result PDF
- Full result report: [`docs/result.pdf`](docs/result.pdf)
- Caption comparison table: [`docs/generated_captions_comparison.csv`](docs/generated_captions_comparison.csv)

## Repository Structure
```text
blip2_lora_finetune/
  build_train_annotations.py      # Build training JSON from Flickr8K text files
  build_test_image_list.py        # Build test image JSON list
  blip2_lora_loader.py            # Load BLIP-2 and attach LoRA adapters
  lora_settings.py                # LoRA hyperparameter settings
  flickr8k_train_dataset.py       # Dataset class for training
  flickr8k_inference_dataset.py   # Dataset class for inference-time image loading
  model_train_utils.py            # Training utilities (freeze/print trainable params)
  train_lora.py                   # Training entry script
  run_inference.py                # Inference entry script

src/
  flickr8k_evaluate.py            # Metric evaluation from JSON predictions

scripts/
  prepare_data.ps1/.sh
  train.ps1/.sh
  infer.ps1/.sh
  eval.ps1/.sh
```


```
## Dataset Preparation
This repo does not ship Flickr8K. Prepare files locally and set paths with environment variables.

Required:
- Image folder (for example: `Flickr8k_Dataset` or `Flicker8k_Dataset`)
- Text files including `Flickr8k.token.txt` and `Flickr_8k.trainImages.txt`

Windows CMD example:
```cmd
set FLICKR8K_TEXT_DIR=D:\path\to\flickr8k_text
set FLICKR8K_IMAGE_DIR=D:\path\to\Flickr8k_Dataset
set FLICKR8K_TRAIN_JSON=blip2_lora_finetune\flickr8k_train.json
set FLICKR8K_TEST_JSON=blip2_lora_finetune\flickr8k_test.json
```

Linux/macOS example:
```bash
export FLICKR8K_TEXT_DIR=/path/to/flickr8k_text
export FLICKR8K_IMAGE_DIR=/path/to/Flickr8k_Dataset
export FLICKR8K_TRAIN_JSON=blip2_lora_finetune/flickr8k_train.json
export FLICKR8K_TEST_JSON=blip2_lora_finetune/flickr8k_test.json
```

## Script-First Run (Recommended)
### Windows CMD
Run PowerShell scripts from CMD:
```cmd
powershell -ExecutionPolicy Bypass -File scripts\prepare_data.ps1
powershell -ExecutionPolicy Bypass -File scripts\train.ps1
powershell -ExecutionPolicy Bypass -File scripts\infer.ps1
powershell -ExecutionPolicy Bypass -File scripts\eval.ps1
```

### PowerShell
```powershell
.\scripts\prepare_data.ps1
.\scripts\train.ps1
.\scripts\infer.ps1
.\scripts\eval.ps1
```

### Linux/macOS
```bash
bash scripts/prepare_data.sh
bash scripts/train.sh
bash scripts/infer.sh
bash scripts/eval.sh
```

## Direct Python Entry Points
```bash
python blip2_lora_finetune/build_train_annotations.py
python blip2_lora_finetune/build_test_image_list.py
python blip2_lora_finetune/train_lora.py
python blip2_lora_finetune/run_inference.py
python src/flickr8k_evaluate.py
```


## Optimization Techniques
Current acceleration/efficiency techniques used in this project:

- Optional 8-bit model loading (`LOAD_IN_8BIT=1`) for lower memory usage.
- Automatic device placement via `device_map="auto"`.
- Mixed precision training (`fp16`) on CUDA.
- Mixed precision inference with `torch.amp.autocast` on CUDA.
- LoRA adapter merge during inference (`merge_and_unload`) to reduce runtime overhead.

Configuration knobs:
- `LOAD_IN_8BIT`
- `TRAIN_FP16`
- `BLIP2_BASE_MODEL`
- `LORA_CHECKPOINT`

## Runtime Configuration
Common environment variables:
- `BLIP2_BASE_MODEL` (default: `Salesforce/blip2-opt-2.7b`)
- `BLIP2_TOKENIZER_PATH`
- `BLIP2_IMAGE_PROCESSOR_PATH`
- `BLIP2_PROCESSOR_PATH`
- `LOAD_IN_8BIT` (`1` or `0`)
- `LORA_CHECKPOINT`
- `INFER_OUTPUT_JSON`
- `EVAL_TEST_JSON`
- `EVAL_PRED_JSON`
- `EVAL_OUTPUT_JSON`

## Notes
- `flickr8k_test.json` is an inference list by default and may contain empty captions. Use a reference-caption JSON for BLEU/ROUGE/METEOR/CIDEr evaluation.
- Full `Salesforce/blip2-opt-2.7b` training needs substantial GPU memory and disk cache.
