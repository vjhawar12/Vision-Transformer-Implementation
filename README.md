# Vision Transformer (ViT) - From Scratch

## 🧠  Motivation: Why Vision Transformers?

Convolutional Neural Networks (CNNs) have long been the dominant architecture for image classification. They are not only highly data-efficient and lightweight—consider MobileNet or EfficientNet—but also highly accurate: for example, ResNet-50 achieves around 95% accuracy on CIFAR-10.

However, the introduction of Vision Transformers (ViTs) brought a new way of thinking to the field—one that applies the attention mechanism first proposed in the 2017 paper “Attention Is All You Need.”

I took on this project to deepen my understanding of attention-based models and explore how these techniques can be applied beyond text data, specifically to vision tasks like image classification.

## 🔍 Overview

This project is a PyTorch-based implementation of the paper “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.” It builds the Vision Transformer (ViT) architecture from scratch, including:

**Patch embeddings**

**Multi-head self-attention**

**Transformer encoders**

The training pipeline also integrates:

**Regularization: stochastic depth, dropout**

**Data augmentation: CutMix, MixUp, horizontal flip, and color jitter**

The model is trained and evaluated on the CIFAR-10 dataset, achieving a final test accuracy of **85.7%**.

## 🛠️ Dependency installation
Clone the repo:

```bash
git clone https://github.com/vjhawar12/Vision-Transformer-paper-implementation.git
cd Vision-Transformer-paper-implementation
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Running the Notebook

Option A: Open the notebook:

```bash
jupyter notebook Vision_Transformer_from_scratch.ipynb
```

Option B: Use Google Colab.

          Run all cells sequentially.
