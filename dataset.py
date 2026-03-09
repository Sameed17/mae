import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

from config import IMAGE_SIZE


class TinyImageNet(Dataset):
    """
    Load images from Tiny ImageNet-200.
    Expects folder structure:
      tiny-imagenet-200/train/n01443507/images/*.JPEG
      tiny-imagenet-200/val/images/*.JPEG
    """

    def __init__(self, root, split="train"):
        """
        root: path to tiny-imagenet-200 folder
        split: "train" or "val"
        """
        self.root = Path(root)
        self.split = split
        self.samples = []

        if split == "train":
            # Each class has an images/ subfolder
            for class_dir in (self.root / "train").iterdir():
                if not class_dir.is_dir():
                    continue
                img_dir = class_dir / "images"
                if img_dir.exists():
                    for p in img_dir.glob("*.JPEG"):
                        self.samples.append(str(p))
        else:
            # Val: images are in val/images/
            img_dir = self.root / "val" / "images"
            if img_dir.exists():
                self.samples = [str(p) for p in img_dir.glob("*.JPEG")]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img
