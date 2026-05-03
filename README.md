# Brain Tumor Detection & Classification Using Deep Learning

> An ensemble deep learning system for automated brain tumor classification from MRI scans, with Explainable AI (Grad-CAM) integration.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red?logo=pytorch)
![Accuracy](https://img.shields.io/badge/Ensemble%20Accuracy-97.6%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

This project presents a comprehensive deep learning framework for **detecting and classifying brain tumors** from MRI images into four categories:

| Class | Description |
|---|---|
|  Glioma | Tumor arising from glial cells |
| Meningioma | Tumor from meninges (brain lining) |
|  Pituitary | Tumor in the pituitary gland |
|  No Tumor | Healthy brain |

The system combines **transfer learning**, **fine-tuning**, **ensemble methods** (soft voting + stacking), and **Grad-CAM** explainability into a unified clinical decision-support framework.

---

## System Architecture

```
Dataset (7,200 MRI Images)
        │
        ▼
Preprocessing & Augmentation
(Resize → Normalize → Rotate/Flip/Zoom)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  CNN Model 1                        CNN Model 2 ... N
(Transfer Learning + Fine-Tune)
        │
        ▼
  Predictions (Probability Distributions)
        │
        ├──────────────────┐
        ▼                  ▼
  Soft Voting          Stacking
  (Average P)     (Logistic Regression)
        │
        ▼
  Ensemble Classification
        │
        ▼
  Grad-CAM XAI Visualization
        │
        ▼
  Diagnosis Decision
```

---

## 📊 Results

### Transfer Learning vs Fine-Tuning (Test Accuracy)

| Model | Transfer Learning | Fine-Tuning |
|---|---|---|
| MobileNetV2 | 89.25% | 92.10% |
| DenseNet-121 | 89.38% | 90.69% |
| InceptionV3 | 89.38% | 91.56% |
| **VGG-16** | 86.37% | **94.69%** |
| ResNet-50 | 89.19% | 93.50% |
| EfficientNet-B3 | 91.25% | 91.87% |

### Ensemble Performance

| Metric | Soft Voting | Stacking |
|---|---|---|
| Accuracy | 97.10% | **97.60%** |
| Precision | 96.5% | 97.0% |
| Recall | 95.5% | 96.0% |
| F1 Score | 95.9% | 96.49% |

> **Key Finding:** Stacking improves over soft voting by <1%, making soft voting the preferred approach due to its lower complexity.

---

## Dataset

- **Total Images:** 7,200 MRI scans
- **Classes:** 4 (Glioma, Meningioma, Pituitary, No Tumor)
- **Train / Test Split:** 5,600 / 1,600 (1,400 & 400 per class)
- **Validation:** 80:20 split from training set

**Sources:**
- [Figshare Brain MRI Dataset](https://figshare.com/)
- [SARTAJ Dataset](https://www.kaggle.com/)
- [Br35H Dataset](https://www.kaggle.com/)

---

## Getting Started

### Prerequisites

```bash
Python 3.10+
CUDA-compatible GPU (recommended)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection

# Install dependencies
pip install -r requirements.txt
```

### Project Structure

```
brain-tumor-detection/
├── dataset/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── pituitary/
│   │   └── notumor/
│   └── Testing/
│       └── ...
├── models/
│   ├── transfer_learning.py
│   ├── fine_tuning.py
│   └── ensemble.py
├── explainability/
│   └── gradcam.py
├── notebooks/
│   └── experiments.ipynb
├── results/
│   ├── training_graphs/
│   └── gradcam_heatmaps/
├── requirements.txt
└── README.md
```

### Training

```bash
# Train individual CNN models (transfer learning)
python models/transfer_learning.py --model densenet121 --epochs 100

# Fine-tune a pretrained model
python models/fine_tuning.py --model vgg16 --lr 1e-5 --epochs 100

# Run ensemble (soft voting)
python models/ensemble.py --strategy soft_voting

# Run ensemble (stacking)
python models/ensemble.py --strategy stacking
```

### Generate Grad-CAM Heatmaps

```bash
python explainability/gradcam.py --image path/to/mri.jpg --model vgg16
```

---

##  Methodology

### 1. Preprocessing Pipeline
- **Resize** — Images resized to model-specific input dimensions
- **Normalize** — Channel-wise mean subtraction (ImageNet stats)
- **Augment** — Rotation (±15°), zoom (15%), shift (10%), horizontal flip

### 2. Transfer Learning
Pretrained CNNs (ImageNet weights) used as feature extractors with frozen base layers. Custom classification head added:
```
Base CNN (frozen) → GlobalAveragePooling → BatchNorm → Dense(256, ReLU) → Dropout(0.4) → Softmax(4)
```

### 3. Fine-Tuning
Final convolutional block unfrozen and retrained at LR = 1×10⁻⁵ with weight decay regularization.

### 4. Ensemble Learning
- **Soft Voting:** `P_final = (P₁ + P₂ + P₃) / 3`
- **Stacking:** Logistic Regression meta-learner trained on base model probability outputs

### 5. Explainability (Grad-CAM)
Gradient-weighted Class Activation Maps highlight the MRI regions most influential in the model's prediction. Red areas = high attention, Blue areas = low attention.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow/Keras, PyTorch |
| ML Utilities | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Platform | Google Colab / VS Code |
| Hardware | Apple M2 (8-core GPU), 16GB RAM |

---

## 📉 Training Graphs

Training and validation accuracy/loss curves are available in `results/training_graphs/` for all models under both transfer learning and fine-tuning strategies.

---

## 🔮 Future Work

- [ ] Expand dataset from multi-hospital / multi-device sources
- [ ] Integrate clinical metadata (patient records + radiology reports)
- [ ] Add tumor segmentation & localization (bounding boxes)
- [ ] Deploy as a real-time REST API / clinical decision tool
- [ ] Explore transformer-based architectures (ViT, Swin Transformer)
- [ ] Investigate federated learning for privacy-preserving training

---

## References

1. Pereira et al. — Brain Tumor Segmentation Using CNNs in MRI Images, *IEEE Trans. Medical Imaging*, 2016
2. Filatov & Yar — Brain Tumor Diagnosis via Pre-Trained CNNs, *arXiv:2208.00768*, 2022
3. Simonyan & Zisserman — VGGNet, *arXiv:1409.1556*, 2014
4. Szegedy et al. — GoogLeNet/Inception, *CVPR*, 2015
5. He et al. — ResNet, *CVPR*, 2016
6. Huang et al. — DenseNet, *CVPR*, 2017
7. Howard et al. — MobileNets, *arXiv:1704.04861*, 2017
8. Tan & Le — EfficientNet, *ICML*, 2019
9. Selvaraju et al. — Grad-CAM, *ICCV*, 2017
10. Iftikhar et al. — Explainable CNN for Brain Tumor Detection, *Brain Informatics*, 2025

---

---

<p align="center">Made with ❤️ for better medical AI</p>





# Multimodal-AI-Based-Clinical-Decision-Support-System-
Repository containing the google colab link of the work
Model 

MobileNet V2 - https://colab.research.google.com/drive/1gtB_cSrS2UPPQxYfMWaZE5xkuLRVJ7l9
 
Densenet 121 - https://colab.research.google.com/drive/1KmhVVjRuPTJLPqHmjGEZyd9vLu8OsL4F#scrollTo=wXBdqp7SyBQo

Inception V3 - https://colab.research.google.com/drive/1n2nJZ7COQyLTBol-yqQabPSTxlZGeqA_#scrollTo=67iPOIywdvI1

Vgg 16 - https://colab.research.google.com/drive/1k98GIQ2T6bJhAmFsECqiknk6jJHkwl4s#scrollTo=BqfRwX4Vsjcq

Resnet 50 - https://colab.research.google.com/drive/1mvYDByhOtXkvg64kznWRgrtx-6CKYd3U#scrollTo=c_LCPS9Iwrip

Efficient Net B3 - https://colab.research.google.com/drive/1Glo-7El74DPQZU192gyXr-1RcC3WjEA7#scrollTo=QiLDeeSCtgwT
