# 🫁 Chest X-Ray Pathology Classification

**Course: Deep Learning & Generative AI (DL-GenAI) · IIT Madras**

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange) ![GPU](https://img.shields.io/badge/Hardware-GPU-green)

> ⚠️ This project was developed as part of a college assignment and is currently private for evaluation. The Kaggle notebook will be made public soon. The `.ipynb` file and submission are shared here in the meantime.

---

## 📋 Overview

Chest X-ray imaging is one of the most widely used diagnostic tools for thoracic disease. Radiologists analyze these images to detect conditions ranging from lung infections and fluid accumulation to collapsed lungs and cardiac enlargement. Interpreting radiographs accurately requires deep expertise and becomes error-prone at scale.

This project tackles a **multi-class chest X-ray pathology classification task** across 20 thoracic conditions, built around a clinically-motivated evaluation metric: one that penalizes missed diagnoses far more heavily than false alarms. The modeling choices — per-class bias tuning, WeightedRandomSampler, stratified CV — are all driven by this asymmetry.

---

## 🗂️ Dataset

| Split | Samples | Columns |
|-------|---------|---------|
| Train | 51,043 | 21 (id + 20 class labels) |
| Test | 17,015 | 1 (id) |

**Pathology Classes (20 total):**
Atelectasis · Cardiomegaly · Consolidation · Edema · Effusion · Emphysema · Fibrosis · Hernia · Infiltration · Mass · Nodule · Pleural_Thickening · Pneumonia · Pneumothorax · Pneumoperitoneum · Pneumomediastinum · Subcutaneous Emphysema · Tortuous Aorta · Calcification of the Aorta · No Finding

---

## 📐 Evaluation Metric

The competition uses a **macro-averaged asymmetric cost function** that reflects real-world clinical stakes:

| Prediction Outcome | Description | Score |
|-------------------|-------------|-------|
| True Positive (TP) | Correctly predicting a disease | +1 |
| False Positive (FP) | Predicting a disease when absent | −1 |
| False Negative (FN) | Failing to detect a disease | **−5** |

**Why FN is penalised 5×:** A false negative sends a patient home with an undetected disease — pneumonia goes untreated, a mass goes uninvestigated. A false positive, by contrast, results in a follow-up scan: costly and stressful, but recoverable. The asymmetric penalty encodes this clinical reality directly into the optimization target: the model must lean toward recall over precision, especially for rare pathologies.

Class-level score:

$$\text{Score}_c = \frac{TP_c - FP_c - 5 \cdot FN_c}{N_c}$$

Final score (macro-average across all C classes):

$$\text{Final Score} = \frac{1}{C} \sum_{c=1}^{C} \text{Score}_c$$

Macro-averaging ensures every pathology — including rare ones — contributes equally, preventing models from gaming the metric by over-predicting the dominant `No Finding` class.

**Reading the scores:** The metric is competition-specific and scores are bounded below by −5 (all predictions wrong) and at +1 (perfect). A naive baseline of always predicting `No Finding` scores approximately **−4.90** on the validation set — since every actual disease case becomes a false negative (penalty: −5). The tuned model's mean validation score of **−4.47** represents a meaningful improvement over this baseline, achieved by correctly surfacing rare pathologies that the naive model suppresses entirely.

---

## 🏗️ Model Architecture & Approach

### Backbone
DenseNet-121 pretrained on ImageNet, with the final classifier replaced by a linear layer mapping to 20 classes.

### Data Preprocessing & Augmentation
- Images resized to 224×224
- Training augmentations: random horizontal flip, random rotation (±15°), color jitter (brightness, contrast, saturation, hue)
- Normalization with ImageNet mean and std
- Class imbalance handled via `WeightedRandomSampler` — without this, the model collapses to predicting `No Finding` for almost everything, since it dominates the training set by a wide margin

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

After training each fold, per-class logit biases are tuned greedily on the validation set to **directly maximize the asymmetric competition score** — not standard accuracy or F1. This step is the most important modeling decision in the pipeline: standard cross-entropy training optimizes for a symmetric loss, but the competition penalizes false negatives 5× more than false positives. Without calibration, the model's decision boundary sits in the wrong place for nearly every class.

---

## 📊 Results

| Fold | Best Epoch | Raw Val Score | Tuned Val Score |
|------|-----------|---------------|-----------------|
| 1 | 4 | −4.6928 | −4.4166 |
| 2 | 2 | −4.6179 | −4.5330 |
| 3 | 4 | −4.6601 | −4.4490 |
| **Mean** | — | **−4.6569** | **−4.4662** |

**How to read these numbers:** Scores range from −5 (worst) to +1 (perfect) under this competition metric. A naive baseline of always predicting `No Finding` scores ~−4.90. The raw trained model (before bias tuning) already improves on this to −4.66 by learning to surface some disease classes. Bias tuning pushes this further to −4.47, a gain of ~0.19 points — meaningful at this scale, representing correctly reclassifying a non-trivial number of disease cases that were previously suppressed.

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

1. Clone the repo and open the notebook on Kaggle or locally.
2. Attach the competition dataset from the Kaggle competition page.
3. Ensure a GPU runtime is selected (Tesla T4 recommended).
4. Run all cells sequentially — the notebook handles training, bias tuning, and submission assembly end-to-end.

**Requirements:**
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

- **Asymmetric losses demand calibration.** Standard cross-entropy training optimizes the wrong objective. Post-hoc bias tuning against the actual competition metric — not a proxy — yields +0.19 points of measurable improvement.
- **WeightedRandomSampler is load-bearing.** Without it, the model collapses to predicting `No Finding` overwhelmingly, maximizing nominal accuracy while scoring near the naive baseline on the actual metric.
- **Macro-averaging is a strict evaluator.** Rare pathologies like Hernia and Pneumomediastinum carry equal weight as common ones like Effusion. The model must generalize across all 20 classes, not just the frequent ones.
- **Baseline anchoring matters.** The naive "always No Finding" baseline (~−4.90) is the right reference point for interpreting results — not zero, not −5.

---

*IITM BS Degree Program · Roll No. 22f3002870 · Parkhi Yadav*
