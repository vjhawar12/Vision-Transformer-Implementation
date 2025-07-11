# Vision Transformer (ViT) - From Scratch

## 🧠  Motivation: Why Vision Transformers?
Convolutional Neural Networks have traditionally dominated image classification tasks due to their strong inductive biases which enable efficient learning from relatively small datasets. Architectures like MobileNet and EfficientNet push this efficiency even further, while deeper models like ResNet-50 consistently achieve high accuracy, reaching ~95% on benchmarks such as CIFAR-10.

Vision Transformers (ViTs), by contrast, take a fundamentally different approach. Instead of using convolutions, they model images as sequences of patches and apply self-attention to capture long-range dependencies across the entire image. Introduced in "Attention Is All You Need", the attention mechanism allows ViTs to learn more flexible, global representations. They, do, however, require lots of data and compute to perform well. 

I chose to implement a ViT to explore how attention can be applied to tasks outside natural language processing--specifically, computer vision. I was curious to learn how shifting my perspective from convolutions to attention offers a different set of trade-offs in terms of scalability and data requirements, but also the potential for richer feature representations.


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


## Results ##
- CIFAR‑10 test accuracy: 85.7%
- Regularization: MixUp, CutMix, stochastic depth
