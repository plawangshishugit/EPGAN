import os
from torch.utils.data import Dataset
from PIL import Image

class EUVPDataset(Dataset):
    """
    EUVP Paired Dataset Loader

    Expected Structure:
        /EUVP/Paired/
            underwater_dark/trainA
            underwater_dark/trainB
            underwater_imagenet/trainA
            underwater_imagenet/trainB
            underwater_scenes/trainA
            underwater_scenes/trainB

    Returns (distorted, clean) pairs.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        subsets = ["underwater_dark", "underwater_imagenet", "underwater_scenes"]
        self.pairs = []

        for sub in subsets:
            dist_dir = os.path.join(root_dir, sub, "trainA")
            clean_dir = os.path.join(root_dir, sub, "trainB")

            if not os.path.exists(dist_dir) or not os.path.exists(clean_dir):
                print(f"[Warning] Missing subset folders: {sub}")
                continue

            dist_imgs = sorted(os.listdir(dist_dir))
            clean_imgs = sorted(os.listdir(clean_dir))

            for d, c in zip(dist_imgs, clean_imgs):
                self.pairs.append(
                    (os.path.join(dist_dir, d),
                     os.path.join(clean_dir, c))
                )

        print(f"[EUVP] Total paired samples: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        dist_path, clean_path = self.pairs[idx]

        dist = Image.open(dist_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")

        if self.transform:
            dist = self.transform(dist)
            clean = self.transform(clean)

        return dist, clean
