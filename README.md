## 🚀 How to Run

### **1. Install Dependencies**

Your code requires the following libraries:

```
pip install torch torchvision torchmetrics scikit-image tqdm pillow matplotlib numpy scikit-learn torchsummary seaborn pandas
```

✅ CUDA is automatically detected in your script
✅ No additional configuration required

---

### **2. Dataset Setup (No Files Included)**

This repository **does not contain any dataset files**.

Download datasets only from official sources:

* **EUVP (paired subset only):**
  [https://irvlab.cs.umn.edu/resources/euvp-dataset](https://irvlab.cs.umn.edu/resources/euvp-dataset)

* **UIEB Benchmark (Raws and References only):**
  [https://li-chongyi.github.io/proj_benchmark.html](https://li-chongyi.github.io/proj_benchmark.html)

#### Folder structure expected by the code:

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
```

Update the dataset path in your code here:

```python
dataset = EUVPDataset(
    r'C:\Users\plawa\anaconda3\envs\underwater_gan_new\EUVP\Paired',
    transform=transform
)
````
### **3. Run Training**

Your script already contains the full training loop:

```
python your_script_name.py
```

It will automatically:

✔ split the dataset into train/test
✔ train the generator & discriminator
✔ log metrics to `training_log.csv`
✔ save best models as:

```
generator_best.pth
discriminator_best.pth
```

✔ generate visual outputs like:

```
test_results.png
edge_analysis_paper_ready_with_metrics.png
col1_gan_loss_curve.png
```

---

## ✅ Reproduction Steps

To reproduce reported results:

1. Download EUVP paired dataset from the official link
2. Place it under `/EUVP/Paired/` as shown above
3. Install requirements
4. Run the script
5. The code will automatically:

   * compute PSNR & SSIM per epoch
   * save best checkpoint
   * evaluate on test split
   * generate publication-ready figures
6. Final metrics will print at the end:

```
PSNR: XX.XX ± XX.XX
SSIM: 0.9XXX ± 0.0XXX
```

---

## 🔗 Dataset Access (No Redistribution)

This project **does not redistribute** any dataset.

Users must download datasets directly from:

* EUVP Dataset — official source only
* UIEB Benchmark — official source only

All rights remain with the original dataset authors.
