# chest-xray-pathology-classification
Deep learning model for multi-class classification of thoracic diseases from chest X-ray images, addressing class imbalance and asymmetric error costs.


# 🫁 Chest X-Ray Pathology Classification

<div align="center">

**Course:** Deep Learning & Generative AI (DL-GenAI) &nbsp;|&nbsp; **Institution:** IIT Madras

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/parkhiyadav/chest-xray-pathology-classification)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![GPU](https://img.shields.io/badge/GPU-Tesla%20T4-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://www.nvidia.com)

🔗 **[View Kaggle Notebook / Competition](https://www.kaggle.com/competitions/26-t-1-dl-gen-ainppe-1)**

</div>

---

## 📋 Overview

Chest X-ray imaging is one of the most widely used diagnostic tools for identifying thoracic diseases. Radiologists analyze these images to detect abnormalities such as lung infections, fluid accumulation, lung collapse, and heart enlargement. However, interpreting chest radiographs requires deep expertise and is time-consuming at scale.

This project addresses a **multi-class chest X-ray pathology classification** task, where each image is assigned to exactly one pathology class from a set of 20 thoracic conditions. The goal is to build a deep learning model that accurately identifies diseases from chest radiographs, with a clinically-motivated scoring function that penalizes missed diagnoses far more heavily than false alarms.

---

## 🗂️ Dataset

| Split | Samples | Columns |
|-------|---------|---------|
| Train | 51,043 | 21 (id + 20 class labels) |
| Test | 17,015 | 1 (id) |

### Pathology Classes (20 total)

`Atelectasis` · `Cardiomegaly` · `Consolidation` · `Edema` · `Effusion` · `Emphysema` · `Fibrosis` · `Hernia` · `Infiltration` · `Mass` · `Nodule` · `Pleural_Thickening` · `Pneumonia` · `Pneumothorax` · `Pneumoperitoneum` · `Pneumomediastinum` · `Subcutaneous Emphysema` · `Tortuous Aorta` · `Calcification of the Aorta` · `No Finding`

---

## 📐 Evaluation Metric

The competition uses a **macro-averaged asymmetric cost function** designed to reflect real-world clinical stakes. Missing a disease is penalized far more harshly than a false alarm.

| Prediction Outcome | Description | Score |
|--------------------|-------------|-------|
| True Positive (TP) | Correctly predicting a disease | **+1** |
| False Positive (FP) | Predicting a disease when it is absent | **−1** |
| False Negative (FN) | Failing to detect a disease | **−5** |

**Class-level score** for pathology *c*:

$$\text{Score}_c = \frac{TP_c - FP_c - 5 \cdot FN_c}{N_c}$$

**Final competition score** (macro-average across all *C* classes):

$$\text{Final Score} = \frac{1}{C} \sum_{c=1}^{C} \text{Score}_c$$

> Macro-averaging ensures every pathology — including rare ones — contributes equally, preventing models from gaming the metric by over-predicting `No Finding`.

---

## 🏗️ Model Architecture & Approach

### Backbone
- **DenseNet-121** pretrained on ImageNet, with the final classifier replaced by a linear layer mapping to 20 classes.

### Data Preprocessing & Augmentation
- Images resized to **224×224**
- Training augmentations: random horizontal flip, random rotation (±15°), color jitter (brightness, contrast, saturation, hue)
- Normalization with ImageNet mean and std
- Class-imbalance handled via **WeightedRandomSampler**

### Training Strategy — 3-Fold Stratified Cross-Validation

| Configuration | Value |
|---------------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Loss Function | Cross-Entropy |
| Batch Size | 48 |
| Epochs per Fold | 5 |
| Hardware | NVIDIA Tesla T4 (GPU) |

### Post-Training Threshold Tuning (Bias Calibration)
After training each fold, **per-class logit biases** are tuned greedily on the validation set to maximize the asymmetric competition score. This step explicitly optimizes for the clinical penalty function rather than standard accuracy.

---

## 📊 Results

| Fold | Best Epoch | Raw Val Score | Tuned Val Score |
|------|------------|---------------|-----------------|
| 1 | 4 | −4.6928 | −4.4166 |
| 2 | 2 | −4.6179 | −4.5330 |
| 3 | 4 | −4.6601 | −4.4490 |
| **Mean** | — | **−4.6569** | **−4.4662** |

> Bias tuning improved the mean validation score by ~**0.19 points**, demonstrating meaningful gains from calibrating predictions to the asymmetric loss.

### Final Submission Prediction Distribution

```
No Finding               13,290
Effusion                  1,105
Infiltration                975
Atelectasis                 612
Pneumothorax                462
Mass                        236
...
```

---

## 🔁 Pipeline Summary

```
Raw Images + CSVs
       │
       ▼
  Data Loading & EDA
       │
       ▼
  Stratified K-Fold Split (k=3)
       │
  ┌────┴────┐
  │  Train  │  ──── DenseNet-121 ──── Cross-Entropy Loss ──── AdamW
  └────┬────┘
       │
  Validation Inference
       │
  Per-Class Bias Tuning (greedy, asymmetric score)
       │
  Test Inference (best epoch weights)
       │
  Ensemble (average logits across folds)
       │
  Argmax → One-Hot → submission.csv
```

---

## 📁 Repository Structure

```
chest-xray-pathology-classification/
├── chest-xray-pathology-classification.ipynb   # Main notebook
├── submission.csv                               # Final predictions
└── README.md                                    # This file
```

---

## 🚀 How to Run

1. **Clone the repo** and open the notebook on Kaggle or locally.
2. **Attach the competition dataset** from the [Kaggle competition page](https://www.kaggle.com/competitions/26-t-1-dl-gen-ainppe-1).
3. Ensure a **GPU runtime** is selected (Tesla T4 recommended).
4. Run all cells sequentially — the notebook handles training, bias tuning, and submission assembly end-to-end.

### Requirements

```
torch>=2.10
torchvision
pandas
numpy
scikit-learn
Pillow
matplotlib
```

---

## 🔑 Key Takeaways

- **Asymmetric losses demand calibration** — standard cross-entropy training alone is suboptimal; post-hoc bias tuning against the actual competition metric yields measurable improvements.
- **WeightedRandomSampler** helps surface rare pathologies during training, preventing the model from collapsing to the dominant `No Finding` class.
- **Macro-averaging** is a strict evaluator: models must generalize across all 20 classes, not just common ones.

---

<div align="center">

*Submitted as part of the DL-GenAI course project at IIT Madras*

**Parkhi Yadav**

</div>
