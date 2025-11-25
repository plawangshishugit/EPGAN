import torch
import numpy as np
import cv2

def get_edge_map(img):
    """
    EXACT user logic:
    - img in [-1,1]
    - convert to [0,255]
    - convert to grayscale
    - apply Canny
    - return (B,1,H,W)
    """
    img_np = (img + 1) / 2     # [0,1]
    img_np = img_np * 255.0
    img_np = img_np.detach().cpu().numpy().astype(np.uint8)

    edges_all = []
    for im in img_np:
        im = np.transpose(im[:3], (1,2,0))  # RGB only
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray, 50, 150)
        edge = edge.astype(np.float32) / 255.0
        edges_all.append(edge)

    edges = np.stack(edges_all, axis=0)   # (B,H,W)
    edges = torch.from_numpy(edges).unsqueeze(1).float()
    return edges.to(img.device)
