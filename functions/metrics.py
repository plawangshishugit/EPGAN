import torch
import numpy as np
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure
)

def psnr_score(fake, real):
    return peak_signal_noise_ratio(fake, real, data_range=2.0).item()

def ssim_score(fake, real):
    return structural_similarity_index_measure(fake, real, data_range=2.0).item()


# ----------------------------------------------------------
# PIXEL-WISE CONFUSION MATRIX (EXACT LOGIC)
# ----------------------------------------------------------
def compute_pixel_confusion(pred_edge, true_edge, threshold=0.1):
    pred = (pred_edge > threshold).float()
    true = (true_edge > threshold).float()

    TP = (pred * true).sum().item()
    FN = ((1 - pred) * true).sum().item()
    FP = (pred * (1 - true)).sum().item()
    TN = ((1 - pred) * (1 - true)).sum().item()

    return TP, FN, FP, TN


# ----------------------------------------------------------
# PRECISION / RECALL / F1 / IoU (EXACT)
# ----------------------------------------------------------
def compute_precision_recall_f1_iou(pred_edge, true_edge, threshold=0.1):
    TP, FN, FP, TN = compute_pixel_confusion(pred_edge, true_edge, threshold)

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = TP / (TP + FP + FN + 1e-8)

    return precision, recall, f1, iou


# ----------------------------------------------------------
# EDGE-FEATURE STATISTICS EXACT AS IN YOUR FILE
# ----------------------------------------------------------
def compute_edge_statistics(pred_edge, true_edge, threshold=0.1):
    TP, FN, FP, TN = compute_pixel_confusion(pred_edge, true_edge, threshold)
    precision, recall, f1, iou = compute_precision_recall_f1_iou(pred_edge, true_edge, threshold)

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    return {
        "TP": TP, "FN": FN, "FP": FP, "TN": TN,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": iou,
        "Accuracy": accuracy,
    }
