import os
from collections import defaultdict
import json

base_dir = "/root/autodl-tmp/data/Flickr8k/captions.txt"
token_path = os.path.join(base_dir, "Flickr8k.token.txt")
train_list_path = os.path.join(base_dir, "Flickr_8k.trainImages.txt")
output_path = os.path.join("/root/autodl-tmp/blip2_lora_finetune", "flickr8k_train.json")

with open(train_list_path, "r") as f:
    train_images = set(line.strip() for line in f.readlines())

image_captions = defaultdict(list)
with open(token_path, "r", encoding="utf-8") as f:
    for line in f:
        image_id, caption = line.strip().split('\t')
        image_name = image_id.split('#')[0]
        if image_name in train_images:
            image_captions[image_name].append(caption)

train_data = []
for image_name, captions in image_captions.items():
    if captions:
        train_data.append({"image": image_name, "caption": captions[0]})

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2)

print(f"✅ Saved {len(train_data)} samples to {output_path}")
