import json
import os

# Input/output path configuration.
train_json_path = os.getenv(
    "FLICKR8K_TRAIN_JSON", os.path.join("blip2_lora_finetune", "flickr8k_train.json")
)
image_dir = os.getenv("FLICKR8K_IMAGE_DIR", "")
if not image_dir:
    default_a = os.path.join("data", "flickr8k", "Flickr8k_Dataset")
    default_b = os.path.join("data", "flickr8k", "Flicker8k_Dataset")
    image_dir = default_a if os.path.isdir(default_a) else default_b
test_json_path = os.getenv(
    "FLICKR8K_TEST_JSON", os.path.join("blip2_lora_finetune", "flickr8k_test.json")
)

# Load image names already used by the training split.
with open(train_json_path, "r") as f:
    train_data = json.load(f)
train_image_set = {item["image"] for item in train_data}

# Enumerate all JPG images and keep only those outside the train split.
all_images = [img for img in os.listdir(image_dir) if img.lower().endswith(".jpg")]
test_images = [img for img in all_images if img not in train_image_set]

# Limit to the first 2000 samples for test generation.
test_images = test_images[:2000]
test_data = [{"image": img, "caption": ""} for img in test_images]

# Save test split JSON.
test_dir = os.path.dirname(test_json_path)
if test_dir:
    os.makedirs(test_dir, exist_ok=True)
with open(test_json_path, "w") as f:
    json.dump(test_data, f, indent=2)

print(f"Built test set with {len(test_data)} images: {test_json_path}")
