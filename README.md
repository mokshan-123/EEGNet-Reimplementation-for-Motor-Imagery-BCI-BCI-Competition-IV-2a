# EEGNet Reimplementation for Motor Imagery BCI (BCI Competition IV-2a)

## Overview

This project presents a **reimplementation of EEGNet** for **motor imagery classification** using the **BCI Competition IV Dataset 2a**.

The work reproduces the architecture proposed in the paper:

> **EEGNet: A Compact Convolutional Neural Network for EEG-based Brain–Computer Interfaces**

and evaluates its performance on **sensorimotor rhythm (SMR) motor imagery decoding**.

The primary goal of this project is to **understand, reproduce, and analyze EEGNet**, while also exploring **implementation differences between TensorFlow and PyTorch frameworks**.

The model is trained to classify **four motor imagery tasks**:

* Left Hand
* Right Hand
* Feet
* Tongue

using EEG signals recorded from **22 EEG electrodes**.

---

## Paper Reference

Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J.

**EEGNet: A Compact Convolutional Neural Network for EEG-based Brain–Computer Interfaces**

Journal of Neural Engineering (2018)

Paper: [https://arxiv.org/abs/1611.08024](https://arxiv.org/abs/1611.08024)

The EEGNet architecture was designed as a **compact CNN capable of learning interpretable EEG features while using significantly fewer parameters than conventional CNN models**.

---

## Project Objectives

The goals of this project were:

1. **Reimplement EEGNet architecture from scratch**
2. **Understand EEG-specific convolutional operations**
3. **Develop a full motor imagery BCI training pipeline**
4. **Compare TensorFlow and PyTorch implementations**
5. **Evaluate performance against the original EEGNet paper**

This project focuses on **engineering a reproducible EEG deep learning pipeline** rather than simply replicating results.

---

## Dataset

**BCI Competition IV – Dataset 2a**

| Property           | Value           |
| ------------------ | --------------- |
| Subjects           | 9               |
| Classes            | 4 motor imagery |
| EEG channels       | 22              |
| Sampling rate      | 250 Hz          |
| Trials per subject | 288             |
| Trial duration     | 7.5 s           |

| Class | Description |
| ----- | ----------- |
| 0     | Left hand   |
| 1     | Right hand  |
| 2     | Feet        |
| 3     | Tongue      |

Only **EEG channels were used**. EOG channels were **removed during preprocessing**.

---

## EEGNet Architecture

The implemented model follows the **EEGNet-8,2 configuration**, commonly used for SMR datasets.

| Parameter     | Value |
| ------------- | ----- |
| F1            | 8     |
| D             | 2     |
| F2            | 16    |
| Kernel length | 32    |
| Dropout       | 0.5   |

Architecture structure:

1. Temporal Convolution
2. Depthwise Spatial Convolution
3. Separable Convolution
4. Classification Layer

EEGNet uses **depthwise and separable convolutions to efficiently learn temporal and spatial EEG features while minimizing the number of parameters**.

---

## Implementation Frameworks

### TensorFlow Implementation

Built using `TensorFlow + Keras`, closely following the **original EEGNet repository provided by the authors**.
```
Result: ~55% classification accuracy
```

This implementation served as a **baseline reproduction of the original architecture**.

---

### PyTorch Implementation

The model was **reimplemented in PyTorch**, allowing more control over the training pipeline and experimentation.
```
Result: ~65% classification accuracy
```

Improvement achieved through:

* Improved preprocessing pipeline
* Structured cross-validation
* Augmentation techniques
* Optimized training configuration

---

## TensorFlow vs PyTorch Comparison

| Aspect               | TensorFlow Implementation | PyTorch Implementation   |
| -------------------- | ------------------------- | ------------------------ |
| Framework            | TensorFlow / Keras        | PyTorch                  |
| Implementation style | High-level API            | Custom training pipeline |
| Training flexibility | Limited                   | High                     |
| Experiment control   | Moderate                  | Full                     |
| Final accuracy       | **~55%**                  | **~65%**                 |

---

## Preprocessing Pipeline

### Channel Selection

Only EEG channels were retained. EOG channels removed. Final input: **22 EEG electrodes**.

### Filtering

Band-pass filter: `4 – 38 Hz`

* Removes low-frequency drift
* Removes high-frequency noise
* Isolates **sensorimotor rhythms**

### Resampling

Original: `250 Hz` → Resampled to: `128 Hz`

### Epoch Extraction

Time window: `0.5 – 2.5 seconds after cue`

Final input shape: `(22 channels, 256 samples)`

### Normalization

Channel-wise **Z-score normalization**:
```
X_norm = (X − mean) / std
```

Statistics calculated **only on training data** to avoid data leakage.

---

## Data Augmentation

Applied **only to training data** in the PyTorch implementation:

* Gaussian noise
* Amplitude scaling
* Temporal masking

---

## Training Strategy

### Cross Validation

`4-fold Stratified Cross Validation`

### Data Split
```
Dataset
 ├── Train/Validation (80%)
 │    └── 4-fold CV
 └── Test (20%)
```

### Optimizer
```
Adam | learning rate = 0.001 | weight decay = 1e-4
```

### Learning Rate Scheduler

`Cosine Annealing` — smoother convergence and improved generalization.

### Loss Function

`CrossEntropyLoss`

### Mixed Precision Training

AMP (Automatic Mixed Precision) enabled for faster training and reduced GPU memory usage.

---

## Training Configuration

| Parameter    | Value           |
| ------------ | --------------- |
| Batch size   | 64              |
| Epochs       | 120             |
| Optimizer    | Adam            |
| LR scheduler | CosineAnnealing |
| Dropout      | 0.5             |
| Augmentation | Enabled         |

---

## Results

| Implementation | Accuracy |
| -------------- | -------- |
| TensorFlow     | ~55%     |
| PyTorch        | ~65%     |

The PyTorch pipeline achieved **better performance due to improved training design and augmentation techniques**.

---

## Comparison with Original EEGNet Paper

| Aspect       | Original Paper         | This Project         |
| ------------ | ---------------------- | -------------------- |
| Framework    | TensorFlow             | TensorFlow + PyTorch |
| Architecture | EEGNet                 | EEGNet               |
| Dataset      | BCI IV-2a              | BCI IV-2a            |
| Channels     | 22 EEG                 | 22 EEG               |
| Resampling   | 128 Hz                 | 128 Hz               |
| Epoch window | 0.5–2.5 s              | 0.5–2.5 s            |
| Training     | Within / cross subject | Subject dependent    |
| Augmentation | Not emphasized         | Implemented          |
| Scheduler    | Not specified          | Cosine LR            |
| Accuracy     | 60-67% (For this data set) | 55–65%               |

---

## Example Training Outputs
```
models/
 ├── best_model.pt
 └── last_model.pt

figures/
 ├── training_curves.png
 └── confusion_matrix.png
```

---


## Acknowledgements

**Dataset:** BCI Competition IV — Graz University of Technology

**EEGNet Authors:** Lawhern et al.

---

## License

MIT License
