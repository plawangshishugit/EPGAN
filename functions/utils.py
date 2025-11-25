import torch
from torchvision.utils import save_image
import numpy as np
import cv2

def normalize_tensor(img):
    """[0,1] → [-1,1]"""
    return (img - 0.5) * 2

def denormalize_tensor(img):
    """[-1,1] → [0,1]"""
    return (img + 1) / 2

def save_image_tensor(tensor, path):
    """Save normalized image tensor EXACTLY like user logic."""
    tensor = denormalize_tensor(tensor)
    save_image(tensor, path)

def save_side_by_side(img1, img2, img3, img4, path):
    """
    EXACT user-like behavior:
    Saves a 2x2 comparative panel: input, edge, restored, GT
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(16,4))

    axs[0].imshow(denormalize_tensor(img1).permute(1,2,0).cpu())
    axs[0].set_title("Input")

    axs[1].imshow(img2.squeeze().cpu(), cmap="gray")
    axs[1].set_title("Edges")

    axs[2].imshow(denormalize_tensor(img3).permute(1,2,0).cpu())
    axs[2].set_title("Output")

    axs[3].imshow(denormalize_tensor(img4).permute(1,2,0).cpu())
    axs[3].set_title("Ground Truth")

    for ax in axs: ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def overlay_edges(img, edge, color=(255,0,0)):
    """
    EXACT user overlay:
    - img: (H,W,3)
    - edge: (H,W)
    - overlays red on edge pixels
    """
    img = (denormalize_tensor(img.permute(1,2,0))).cpu().numpy()
    edge = edge.squeeze().cpu().numpy()

    overlay = img.copy()
    overlay[edge > 0.5] = np.array(color) / 255.0
    return overlay
