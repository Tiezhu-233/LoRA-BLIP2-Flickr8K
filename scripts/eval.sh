#!/usr/bin/env bash
set -euo pipefail

# Evaluate generated captions using the experimental evaluation script.
python src/flickr8k_evaluate.py

echo "Evaluation finished."
