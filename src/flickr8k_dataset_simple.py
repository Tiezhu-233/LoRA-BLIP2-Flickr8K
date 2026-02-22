import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Flickr8kDataset(Dataset):
    """Simple Flickr8k loader that reads image paths from a caption CSV file."""

    def __init__(self, image_dir, caption_path, processor, max_length=40):
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        self.caption_file = self._find_caption_file(caption_path)
        self.captions = pd.read_csv(self.caption_file)

        self.valid_indices = []
        for idx, row in self.captions.iterrows():
            img_path = os.path.join(self.image_dir, row["image"])
            if os.path.exists(img_path):
                self.valid_indices.append(idx)

        print(
            f"Total samples: {len(self.captions)}, "
            f"valid samples: {len(self.valid_indices)}"
        )

    def _find_caption_file(self, caption_path: str) -> str:
        """Resolve a caption file path from either a file or a folder."""
        if os.path.isfile(caption_path):
            return caption_path

        search_files = ["Flickr8k.token.txt", "captions.txt", "annotations.txt"]
        if os.path.isdir(caption_path):
            for file in search_files:
                file_path = os.path.join(caption_path, file)
                if os.path.isfile(file_path):
                    return file_path

            for file in os.listdir(caption_path):
                if "flickr8k" in file.lower() and file.lower().endswith(".txt"):
                    if "token" in file.lower():
                        return os.path.join(caption_path, file)

        raise FileNotFoundError(f"No suitable caption file found in {caption_path}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.captions.iloc[real_idx]

        img_path = os.path.join(self.image_dir, row["image"])
        image = Image.open(img_path).convert("RGB")

        return {
            "image": image,
            "dataset_index": real_idx,
            "image_id": row["image"],
        }

