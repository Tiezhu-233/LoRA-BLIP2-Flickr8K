from torch.utils.data import Dataset
from PIL import Image
import os

class Flickr8kTestDataset(Dataset):
    def __init__(self, json_path, processor, image_root):
        """
        :
            json_path: test split json path
            processor: Blip2Processor
            image_root: image folder path
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

        # Comment translated to English and cleaned.
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        # Comment translated to English and cleaned.
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "image_path": image_path  # Comment translated to English and cleaned.
        }
