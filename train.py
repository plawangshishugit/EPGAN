
import os
import argparse
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

# --- Import your modules EXACTLY ---
from model import EnhancedGenerator, MultiScaleDiscriminator
from database import EUVPDataset
from objectives.losses import compute_total_loss, compute_adversarial_loss


# ------------------------
# ARGUMENT PARSER
# ------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train EP-GAN")
    p.add_argument("--data_root", type=str, required=True,
                  help="Path to EUVP/Paired directory")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr_g", type=float, default=0.0002)
    p.add_argument("--lr_d", type=float, default=0.00002)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--log_csv", type=str, default="training_log.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ------------------------
# CSV LOGGING
# ------------------------
def init_csv_logger(filename="training_log.csv"):
    import csv
    if not os.path.exists(filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "epoch",
                "train_g_loss", "train_d_loss",
                "test_psnr", "test_ssim",
                "best_psnr"
            ])


def append_csv(filename, row):
    import csv
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ------------------------
# MAIN TRAINING FUNCTION
# ------------------------
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # Transform (EXACT from your code)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load Dataset
    dataset = EUVPDataset(args.data_root, transform=transform)

    # Train/Test split (80/20)
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=args.seed)

    train_ds = Subset(dataset, train_idx)
    test_ds = Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Training samples: {len(train_ds)}")
    print(f"Testing samples:  {len(test_ds)}")

    # Models
    generator = EnhancedGenerator().to(device)
    discriminator = MultiScaleDiscriminator().to(device)

    # Optimizers
    import torch.optim as optim
    g_opt = optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    d_opt = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    os.makedirs(args.save_dir, exist_ok=True)
    init_csv_logger(args.log_csv)
    best_psnr = 0.0

    # ------------------------
    # TRAINING LOOP
    # ------------------------
    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()

        ep_g_loss = 0.0
        ep_d_loss = 0.0

        tl = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")

        for i, (distorted, gt) in enumerate(tl):
            distorted = distorted.to(device)
            gt = gt.to(device)

            # --- Train Discriminator every 25 batches ---
            if i % 25 == 0:
                d_opt.zero_grad()
                with torch.no_grad():
                    fake = generator(distorted)
                real_noise = gt + torch.randn_like(gt) * 0.1
                fake_noise = fake.detach() + torch.randn_like(fake) * 0.1

                d_loss = compute_adversarial_loss(discriminator, fake_noise, real_noise)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                d_opt.step()
                ep_d_loss += d_loss.item()

            # --- Train Generator every batch ---
            g_opt.zero_grad()
            fake = generator(distorted)

            g_loss, c_loss, a_loss, p_loss, e_loss = compute_total_loss(
                generator, discriminator, fake, gt
            )
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            g_opt.step()

            ep_g_loss += g_loss.item()

            # Progress bar metrics
            if i % 10 == 0:
                with torch.no_grad():
                    psnr = peak_signal_noise_ratio(fake, gt, data_range=2.0).item()
                    ssim = structural_similarity_index_measure(fake, gt, data_range=2.0).item()
                tl.set_postfix(G=f"{g_loss.item():.4f}", D=f"{d_loss.item():.4f}",
                               PSNR=f"{psnr:.2f}", SSIM=f"{ssim:.4f}")

        # ------------------------
        # TESTING (80/20 split)
        # ------------------------
        generator.eval()
        psnrs, ssims = [], []

        with torch.no_grad():
            for distorted, gt in test_loader:
                distorted = distorted.to(device)
                gt = gt.to(device)
                fake = generator(distorted)

                psnrs.append(peak_signal_noise_ratio(fake, gt, data_range=2.0).item())
                ssims.append(structural_similarity_index_measure(fake, gt, data_range=2.0).item())

        mean_psnr = np.mean(psnrs)
        mean_ssim = np.mean(ssims)

        # Save best checkpoint
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            torch.save(generator.state_dict(), os.path.join(args.save_dir, "generator_best.pth"))
            torch.save(discriminator.state_dict(), os.path.join(args.save_dir, "discriminator_best.pth"))
            print(f"Best model updated at epoch {epoch+1} (PSNR={best_psnr:.3f})")

        # Log CSV
        append_csv(args.log_csv, [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            epoch + 1,
            ep_g_loss / len(train_loader),
            ep_d_loss / max(1, len(train_loader)//25),
            mean_psnr,
            mean_ssim,
            best_psnr
        ])

        print(f"Epoch {epoch+1}: PSNR={mean_psnr:.2f}, SSIM={mean_ssim:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
