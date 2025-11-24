## How to Run

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

✅ **Only `trainA` (distorted images) and `trainB` (reference images) are used in this project.**
✅ No validation or test folders from the original dataset are used.

The dataset is internally divided into an **80/20 split** for training and testing using:

```python
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
```

Update the dataset path in your code here:

```python
dataset = EUVPDataset(
    r'C:\Users\plawa\anaconda3\envs\underwater_gan_new\EUVP\Paired',
    transform=transform
)
```

---

### **3. Run Training**

Your script already contains the full training loop:

```
python euvpcode.py
```
or 
```
python uiebcode.py
```

It will automatically:

✔ split the dataset into **80% training / 20% testing**
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

1. Download the EUVP paired dataset from the official link

2. Place only the `trainA` and `trainB` folders under `/EUVP/Paired/`

3. Install requirements

4. Run the script

5. The code will automatically:

   * compute PSNR & SSIM per epoch
   * save the best checkpoint
   * evaluate on the 20% test split
   * generate publication-ready figures

6. Final metrics will print at the end, for example:

```
PSNR: XX.XX ± XX.XX
SSIM: 0.XXXX ± 0.XXXX
```

---

## 🔗 Dataset Access (No Redistribution)

This project **does not redistribute** any dataset.

Users must download datasets directly from:

* EUVP Dataset — official source only
* UIEB Benchmark — official source only

Only officially downloaded **trainA and trainB** images were used, and solely for training/testing within an 80/20 split.

All rights remain with the original dataset authors.
