Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Evaluate generated captions using the experimental evaluation script.
python src/flickr8k_evaluate.py

Write-Host "Evaluation finished."
