# Sleep Stage Classification using TCN
> **CIE 555 — Neural Networks and Deep Learning**  
> University of Science and Technology, Zewail City · Spring 2026  
> **Status: 🚧 Work in Progress — evaluation results and plots coming soon**

---

## Overview

Automated sleep stage classification from EEG signals using a **Temporal Convolutional Network (TCN)** with dilated causal convolutions and residual connections.

Sleep staging is a clinically critical task — automated classification from EEG enables scalable diagnosis of sleep disorders without expensive manual annotation by sleep specialists.

---

## Dataset

| Property | Details |
|----------|---------|
| Source | [Sleep-EDF Expanded Dataset — PhysioNet 1.0.0](https://www.physionet.org/content/sleep-edfx/1.0.0/) |
| Subjects | 2 (SC4001, SC4002) |
| EEG Channel | Fpz–Cz |
| Sampling Rate | 100 Hz |
| Epoch Length | 30 seconds (3,000 samples) |
| Sleep Stages | Wake, N1, N2, N3, REM |

---

## Pipeline

### Preprocessing
- 30-second epoch segmentation aligned with hypnogram annotations
- **StandardScaler normalization** — fit on training set only (no data leakage)
- **Stratified 70/15/15 train/val/test split** — preserves class distribution across all splits

### Why stratified splitting?
Sleep stages are heavily imbalanced — N2 dominates (~50% of epochs) while N1 is rare (~5%). Random splitting risks placing all rare-stage epochs in a single split. Stratified splitting guarantees proportional representation of all 5 classes everywhere.

---

## Model Architecture

A 3-block TCN with increasing filter sizes and exponentially growing dilation rates:

```
Input (3000, 1)
    ↓
Residual Block 1 — Conv1D(32, k=3, d=1) × 2 + Skip
    ↓
Residual Block 2 — Conv1D(32, k=3, d=2) × 2 + Skip
    ↓
Residual Block 3 — Conv1D(64, k=3, d=4) × 2 + Skip
    ↓
GlobalAveragePooling1D
    ↓
Dense(64, ReLU) → Dropout(0.5)
    ↓
Dense(5, softmax)
```

Each residual block: `Conv1D (causal, dilated) → BatchNorm → ReLU → Dropout(0.2)` × 2 + L2 regularization (λ=0.001) + skip connection.

### Why TCN over LSTM for EEG?

| Property | LSTM | TCN |
|----------|------|-----|
| Temporal causality | ✅ | ✅ (causal conv) |
| Long-range dependencies | Limited by hidden state | ✅ (dilated conv) |
| Parallel training | ❌ (sequential) | ✅ |
| Gradient stability | Prone to vanishing | ✅ (residual connections) |
| Full sequence resolution | ❌ (bottleneck) | ✅ |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam, lr = 5×10⁻⁴ |
| Loss | Categorical cross-entropy |
| Batch size | 32 |
| Max epochs | 20 |
| Early stopping | Patience = 3 (val_loss) |
| L2 regularization | λ = 0.001 |

---

## Expected Results

| Stage | Expected Performance | Reason |
|-------|---------------------|--------|
| N2 | High | Dominant class; distinctive spindles and K-complexes |
| N3 | High | High-amplitude delta waves are highly discriminative |
| Wake | High | High-frequency, low-amplitude pattern is distinctive |
| REM | Moderate | Shares EEG characteristics with Wake |
| N1 | Low | Rare stage; no distinctive EEG marker; transitions between Wake and N2 |

---

## Tech Stack

- **Python 3**
- **TensorFlow / Keras** — TCN implementation
- **MNE** — EEG data loading and processing
- **scikit-learn** — stratified splitting, metrics
- **NumPy, Matplotlib** — data handling and visualization

---

## File Structure

```
├── Sleep_Stage_Classification_using_TCN_with_Stratified_Splitting.ipynb
├── report.pdf
└── README.md
```

---

> Dataset downloads automatically from PhysioNet inside the notebook.

---

## Author

**Ahmed Gamal** — [@AhmedGamal04](https://github.com/AhmedGamal04)
