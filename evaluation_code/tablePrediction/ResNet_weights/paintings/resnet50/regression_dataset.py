import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class RegressionDataset(Dataset):
    def __init__(self, root, ann_file, transform=None, target_dim=1):
        self.root = root
        self.transform = transform
        self.target_dim = target_dim
        self.samples = []

        with open(ann_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                img_path = os.path.join(root, parts[0])
                values = list(map(float, parts[1:]))

                if len(values) != target_dim:
                    raise ValueError(f"Expect {target_dim} values, got {len(values)}")

                self.samples.append((img_path, torch.tensor(values, dtype=torch.float)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, target
