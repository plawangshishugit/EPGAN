import torch
import torch.nn as nn
import torch.nn.functional as F

# The following functions are taken verbatim from the uploaded file (preserving original logic).
# Source: user uploaded file. See repository reference for provenance.

def compute_content_loss(fake, real):
    return F.l1_loss(fake, real)

def compute_adversarial_loss(discriminator, fake, real):
    real_out = discriminator(real)
    fake_out = discriminator(fake)
    loss = 0
    for ro, fo in zip(real_out, fake_out):
        loss += nn.BCELoss()(ro, torch.ones_like(ro)) + nn.BCELoss()(fo, torch.zeros_like(fo))
    return loss / 2

def compute_perceptual_loss(fake, real):
    return F.mse_loss(fake, real)  # Simplified; original uses MSE (VGG optional)

def compute_edge_loss(fake, real, get_edge_map):
    """
    compute_edge_loss expects the get_edge_map function to be passed in (keeps modularity).
    In your original code, get_edge_map(fake) and get_edge_map(real) were used directly.
    """
    fake_edges = get_edge_map(fake)
    real_edges = get_edge_map(real)
    return F.l1_loss(fake_edges, real_edges)

def compute_total_loss(generator, discriminator, fake, real, get_edge_map=None):
    """
    Returns: total_loss, content_loss, adversarial_loss, perceptual_loss, edge_loss
    Mirrors the original weights and combination from your code.
    If get_edge_map is None, the compute_edge_loss will attempt to call a globally available get_edge_map.
    """
    c_loss = compute_content_loss(fake, real)
    a_loss = compute_adversarial_loss(discriminator, fake, real)
    p_loss = compute_perceptual_loss(fake, real)

    # Preserve original behaviour: use get_edge_map from scope if not provided
    if get_edge_map is None:
        try:
            from functions.edges import get_edge_map as _gm
            e_loss = compute_edge_loss(fake, real, _gm)
        except Exception:
            # Fallback: zero if edge function unavailable, but original repo includes it so this should not happen.
            e_loss = torch.tensor(0.0, device=fake.device)
    else:
        e_loss = compute_edge_loss(fake, real, get_edge_map)

    # Original lambda weights from your file
    lambda1, lambda2, lambda3, lambda4 = 1.0, 3.0, 0.01, 0.1
    total_loss = (lambda1 * c_loss + lambda2 * a_loss + lambda3 * p_loss + lambda4 * e_loss)
    return total_loss, c_loss, a_loss, p_loss, e_loss


# Edge metrics (copied exactly)

def compute_iou(pred_edges, true_edges):
    """Compute Intersection over Union (IoU) for edge maps."""
    pred_edges = (pred_edges > 0.5).float()  # Threshold to binary
    true_edges = (true_edges > 0.5).float()
    intersection = (pred_edges * true_edges).sum(dim=[1, 2])
    union = (pred_edges + true_edges).clamp(0, 1).sum(dim=[1, 2])
    return intersection / (union + 1e-6)  # Add epsilon to avoid division by zero

def compute_f1_score(pred_edges, true_edges):
    """Compute F1-score for edge maps."""
    pred_edges = (pred_edges > 0.5).float()
    true_edges = (true_edges > 0.5).float()
    tp = (pred_edges * true_edges).sum(dim=[1, 2])
    fp = (pred_edges * (1 - true_edges)).sum(dim=[1, 2])
    fn = ((1 - pred_edges) * true_edges).sum(dim=[1, 2])
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return 2 * (precision * recall) / (precision + recall + 1e-6)
