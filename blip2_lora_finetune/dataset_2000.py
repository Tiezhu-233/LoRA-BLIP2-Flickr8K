from torch.utils.data import Dataset
from PIL import Image
import os

class Flickr8kTestDataset(Dataset):
    def __init__(self, json_path, processor, image_root):
        """
        参数:
            json_path: str, flickr8k_test.json 文件路径
            processor: transformers 的 Blip2Processor
            image_root: str, 图像根目录（如 /root/autodl-tmp/data/Flickr8k/Images）
        """
        import json
        with open(json_path, "r") as f:
            self.samples = json.load(f)
        self.processor = processor
        self.image_root = image_root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = os.path.join(self.image_root, item["image"])
        assert os.path.exists(image_path), f"Image not found: {image_path}"

        image = Image.open(image_path).convert("RGB")

        # 推理时无 caption，使用占位 prompt
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        # 取出 batch 维度
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "image_path": image_path  # 可用于生成时记录文件名
        }
