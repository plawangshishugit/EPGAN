import os
from torch.utils.data import Dataset
from PIL import Image

class UIEBDataset(Dataset):
    """
    UIEB Benchmark Dataset Loader

    Expected Structure:
        /UIEB/raw/
        /UIEB/reference/

    Returns (raw_image, reference_image) pairs.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        raw_dir = os.path.join(root_dir, "raw")
        ref_dir = os.path.join(root_dir, "reference")

        raw_files = sorted(os.listdir(raw_dir)) if os.path.exists(raw_dir) else []
        ref_files = sorted(os.listdir(ref_dir)) if os.path.exists(ref_dir) else []

        if len(raw_files) != len(ref_files):
            print("[UIEB] Warning: Raw and reference pairs mismatch.")

        self.pairs = [
            (os.path.join(raw_dir, r), os.path.join(ref_dir, gt))
            for r, gt in zip(raw_files, ref_files)
        ]

        print(f"[UIEB] Total paired samples: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        raw_path, ref_path = self.pairs[idx]

        raw = Image.open(raw_path).convert("RGB")
        ref = Image.open(ref_path).convert("RGB")

        if self.transform:
            raw = self.transform(raw)
            ref = self.transform(ref)

        return raw, ref
