
# 📘 EP-GAN: Edge-Preserving Generative Adversarial Network for Underwater Image Restoration

This repository contains the official implementation of **EP-GAN**, an edge-preserving generative adversarial network designed to restore underwater images by enhancing color, contrast, structural details, and edge fidelity.
The model integrates global perceptual learning with local edge-aware guidance for high-quality underwater image restoration.

The complete implementation and training code are publicly available under the DOI required for reproducibility.

---

## 🔗 **Code Availability**

The full source code is archived at:

👉 **[https://doi.org/10.5281/zenodo.17649194](https://doi.org/10.5281/zenodo.17649194)**

---

# 📁 Repository Structure

```
EPGAN/
│
├── model/                # Generator, Discriminator, Residual, Attention & Deformable Blocks
├── database/             # Dataset loader (EUVPDataset, UIEB loader) and preprocessing tools
├── functions/            # Utility functions: edge maps, metrics, helpers, visualizations
├── objectives/           # Loss functions: edge loss, perceptual loss, content loss, GAN loss
│
├── train.py              # Training pipeline with hyperparameters & checkpoint saving
├── test.py               # Testing, evaluation & image generation
│
├── README.md             # Documentation and usage instructions
└── LICENSE               # MIT License (or preferred OSS license)
```

---

# 🚀 How to Run

### **1. Install Dependencies**

```
pip install torch torchvision torchmetrics scikit-image tqdm pillow matplotlib numpy scikit-learn torchsummary seaborn pandas
```

✔ GPU is automatically used if available
✔ No additional configs required

---

## **2. Dataset Setup (No Files Included)**

This repository does **NOT** include dataset files.
All datasets must be downloaded from official sources.

### **Official Dataset Links**

* **EUVP (paired subset only)**
  [https://irvlab.cs.umn.edu/resources/euvp-dataset](https://irvlab.cs.umn.edu/resources/euvp-dataset)

* **UIEB Benchmark (raws + references)**
  [https://li-chongyi.github.io/proj_benchmark.html](https://li-chongyi.github.io/proj_benchmark.html)

---

## 📂 Folder Structure Required

```
/EUVP/Paired/
   underwater_dark/
      trainA/
      trainB/
   underwater_imagenet/
      trainA/
      trainB/
   underwater_scenes/
      trainA/
      trainB/

# UIEB should follow its original structure:
# /UIEB/raw/
# /UIEB/reference/
```

### ✔ Important Notes

* Only **trainA** (distorted images) and **trainB** (reference images) are used.
* For each dataset, an **80% training / 20% testing** split is performed **inside the code**.
* Datasets were **not mixed**; each dataset was used **independently**.

---

# 🧠 Training Strategy (IMPORTANT)

We designed the EP-GAN architecture and trained it **separately on two different datasets**:

### ✔ **1. EUVP (paired) Training**

* Only `trainA` and `trainB`
* Internal **80/20 split** for training/testing
* Trained from scratch

### ✔ **2. UIEB Training**

* Only raw–reference paired images
* Internal **80/20 split** for training/testing
* Trained independently from EUVP

### ❗ No cross-dataset training

### ❗ No dataset mixing

### ❗ Each dataset has its own checkpoints & metrics

This ensures unbiased, dataset-specific performance evaluation.

---

# 🧑‍💻 Training

Run:

```
python train.py
```

This script will:

✔ Load dataset
✔ Split into **80% train / 20% test**
✔ Train Generator & Discriminator
✔ Save best checkpoints:

```
generator_best.pth
discriminator_best.pth
```

✔ Log metrics to:

```
training_log.csv
```

✔ Generate visual outputs:

```
test_results.png
edge_analysis_paper_ready_with_metrics.png
col1_gan_loss_curve.png
```

---

# 🧪 Testing / Evaluation

Run:

```
python test.py
```

Outputs include:

* Restored images
* PSNR, SSIM scores
* Edge comparison visualizations
* Side-by-side results

---

# 🔁 Reproduction Steps

1. Install dependencies
2. Download EUVP or UIEB from official sites
3. Place files in the required folder structure
4. Run:

```
python train.py
```

5. After training:

```
python test.py
```

6. Compare evaluation metrics:

```
PSNR: XX.XX ± XX.XX
SSIM: 0.9XXX ± 0.0XXX
```

---

# 📂 Dataset Access (Ethical Compliance)

This project **does not include or redistribute** any dataset.

Users must download datasets directly from their original authors:

* EUVP dataset (research use only)
* UIEB dataset (academic use only; redistribution prohibited)

All rights remain with their respective creators.

---

# 📝 Citation

If you use this code, model, or datasets, please cite:

**"EP-GAN: An Edge Preserving Generative Adversarial Network for Underwater Image Restoration"**
*The Visual Computer (Springer)*, 2025.

---

# 🔖 BibTeX

```bibtex
@article{EPGAN2025,
  title={EP-GAN: An Edge Preserving Generative Adversarial Network for Underwater Image Restoration},
  journal={The Visual Computer},
  year={2025},
  authors={Plawang Shishu, Sruthi Nair, Mayur Parate, Tausif Diwan, Parul Sahare}
}
```

---

# 📜 License

This project is released under the **MIT License**, enabling free academic and research usage.

---


