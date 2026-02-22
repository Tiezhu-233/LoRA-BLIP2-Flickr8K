Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Optional environment variables:
#   BLIP2_BASE_MODEL
#   BLIP2_PROCESSOR_PATH
#   LORA_CHECKPOINT
#   FLICKR8K_IMAGE_DIR
#   FLICKR8K_TEST_JSON
#   INFER_OUTPUT_JSON

python blip2_lora_finetune/run_inference.py

Write-Host "Inference finished."
