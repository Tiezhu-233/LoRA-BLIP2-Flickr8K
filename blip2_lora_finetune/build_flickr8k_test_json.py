import json
import os
from tqdm import tqdm

# 路径配置
train_json_path = "blip2_lora_finetune/flickr8k_train.json"
image_dir = "/root/autodl-tmp/data/Flickr8k/Images"
test_json_path = "blip2_lora_finetune/flickr8k_test.json"

# 加载训练图像名称集合
with open(train_json_path, "r") as f:
    train_data = json.load(f)
train_image_set = set([item["image"] for item in train_data])

# 获取所有图像
all_images = [img for img in os.listdir(image_dir) if img.lower().endswith(".jpg")]

# 选出不在训练集中的图像（作为测试集）
test_images = [img for img in all_images if img not in train_image_set]

# 取前 2000 张未用图片（或全部）
test_images = test_images[:2000]

# 构造测试集，每张图片配一个 placeholder caption
test_data = [{"image": img, "caption": ""} for img in test_images]

# 保存
os.makedirs(os.path.dirname(test_json_path), exist_ok=True)
with open(test_json_path, "w") as f:
    json.dump(test_data, f, indent=2)

print(f"✅ 测试集构建完成，共 {len(test_data)} 张图像，已保存至 {test_json_path}")
