from torch.utils.data import Dataset
from PIL import Image
import os

class Flickr8kCaptionDataset(Dataset):
    def __init__(self, annotations, processor, image_root):
        self.samples = annotations
        self.processor = processor
        self.image_root = image_root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = os.path.join(self.image_root, item["image"])
        assert os.path.exists(image_path), f"Image not found: {image_path}"

        caption = item["caption"]
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "labels": inputs["labels"]
        }
