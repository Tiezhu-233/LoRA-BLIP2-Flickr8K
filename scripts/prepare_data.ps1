Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Optional environment variables:
#   FLICKR8K_TEXT_DIR
#   FLICKR8K_IMAGE_DIR
#   FLICKR8K_TRAIN_JSON
#   FLICKR8K_TEST_JSON

python blip2_lora_finetune/build_train_annotations.py
python blip2_lora_finetune/build_test_image_list.py

Write-Host "Data preparation finished."
