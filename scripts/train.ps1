Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Optional environment variables:
#   FLICKR8K_TRAIN_JSON
#   FLICKR8K_IMAGE_DIR
#   BLIP2_BASE_MODEL
#   LORA_CHECKPOINT

python blip2_lora_finetune/train_lora.py

Write-Host "Training finished."
