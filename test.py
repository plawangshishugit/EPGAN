
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

# --- Import EP-GAN modules ---
from model import EnhancedGenerator
from database import EUVPDataset
from functions.edges import get_edge_map
from functions.utils import denormalize_tensor
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


# ------------------------
# ARGUMENT PARSER
# ------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Test EP-GAN")
    p.add_argument("--data_root", type=str, required=True,
                  help="Path to EUVP/Paired directory")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--checkpoint", type=str, default="generator_best.pth")
    p.add_argument("--save_results", type=str, default="results")
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ------------------------
# LOAD MODEL
# ------------------------
def load_model(ckpt_path, device):
    gen = EnhancedGenerator().to(device)
    gen.load_state_dict(torch.load(ckpt_path, map_location=device))
    gen.eval()
    return gen


# ------------------------
# MAIN
# ------------------------
def main():
    args = parse_args()
    device = torch.device(args.device)

    # Transform (same as training)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Dataset
    dataset = EUVPDataset(args.data_root, transform=transform)

    # Same 80/20 split as training
    indices = list(range(len(dataset)))
    _, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    test_ds = Subset(dataset, test_idx)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Total test samples: {len(test_ds)}")

    # Load generator
    generator = load_model(args.checkpoint, device)
    os.makedirs(args.save_results, exist_ok=True)

    # -----------------------------
    # Compute PSNR & SSIM
    # -----------------------------
    psnrs, ssims = [], []

    with torch.no_grad():
        for distorted, gt in test_loader:
            distorted = distorted.to(device)
            gt = gt.to(device)

            fake = generator(distorted)

            psnrs.append(peak_signal_noise_ratio(fake, gt, data_range=2.0).item())
            ssims.append(structural_similarity_index_measure(fake, gt, data_range=2.0).item())

    print(f"\nFinal Test PSNR: {np.mean(psnrs):.3f} ± {np.std(psnrs):.3f}")
    print(f"Final Test SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")

    # -----------------------------
    # Visualize random samples
    # -----------------------------
    all_distorted = []
    all_gt = []

    for d, g in test_loader:
        all_distorted.append(d)
        all_gt.append(g)

    all_distorted = torch.cat(all_distorted, dim=0)
    all_gt = torch.cat(all_gt, dim=0)

    total_samples = all_distorted.size(0)
    indices = random.sample(range(total_samples), min(args.num_samples, total_samples))

    fig, axs = plt.subplots(len(indices), 4, figsize=(16, 5 * len(indices)))

    if len(indices) == 1:
        axs = axs.reshape(1, -1)

    for i, idx in enumerate(indices):
        distorted = all_distorted[idx].unsqueeze(0).to(device)
        gt = all_gt[idx].unsqueeze(0).to(device)

        with torch.no_grad():
            fake = generator(distorted)

        # Convert to numpy
        distorted_np = denormalize_tensor(distorted[0, :3]).cpu().permute(1,2,0).numpy()
        fake_np      = denormalize_tensor(fake[0]).cpu().permute(1,2,0).numpy()
        gt_np        = denormalize_tensor(gt[0]).cpu().permute(1,2,0).numpy()
        edge_np      = get_edge_map(distorted)[0, 0].cpu().numpy()

        axs[i,0].imshow(distorted_np)
        axs[i,0].set_title("Distorted Input")

        axs[i,1].imshow(edge_np, cmap="gray")
        axs[i,1].set_title("Canny Edge Map")

        axs[i,2].imshow(fake_np)
        axs[i,2].set_title("Generated Output")

        axs[i,3].imshow(gt_np)
        axs[i,3].set_title("Ground Truth")

        for ax in axs[i]:
            ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(args.save_results, "test_results.png")
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"\nSaved test visualization → {save_path}")

    print("\nTesting complete.")


if __name__ == "__main__":
    main()
