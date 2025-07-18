# 🧠 Spatio-Temporal Activity Recognition via ResNet-BiLSTM and MHSA-Based Transformer Encoders

This project presents a deep learning pipeline for video classification, focusing on **multi-class activity recognition** using **spatio-temporal modeling**. We integrate **ResNet-18** for spatial encoding, **Bidirectional LSTM** for temporal dynamics, and **Multi-Head Self-Attention (MHSA)** with **Transformer Encoder Layers (TEL)** to capture global dependencies across time.

## 📌 Key Highlights

- 🧠 **Hybrid Deep Architecture**: Combines CNN-based spatial extraction, RNN-based sequence modeling, and attention-based context modeling.
- 🎞️ **Custom Video Dataset Handling**: Uses OpenCV for efficient frame sampling, normalization, and augmentation.
- ⚙️ **Training Stability**: Implements gradient clipping, stratified sampling, learning rate scheduling, and early stopping.
- 🏷️ **Class Imbalance Aware**: Dynamic class weighting during loss computation for skewed label distributions.
- 📊 **Rich Evaluation**: Classification report and confusion matrix visualization for validation insights.

---

## Results
![0 86](https://github.com/user-attachments/assets/7a921c0e-952d-42eb-9a18-6d7ba79c0a3a)


## 📁 Architecture Overview

```text
           Input Video
               ↓
         OpenCV Frame Sampling
               ↓
          ResNet-18 Backbone
               ↓
        BiLSTM Temporal Encoder
               ↓
       Multi-Head Self-Attention
               ↓
       Transformer Encoder Layers
               ↓
       Global Average Pooling
               ↓
          Classification Head
               ↓
           Output Class
