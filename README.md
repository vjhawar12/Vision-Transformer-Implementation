# Vision Transformer (ViT) - From Scratch

## 🧠  Motivation: Why Vision Transformers?

Convolutional Neural Networks (CNNs) have traditionally dominated image classification tasks due to their strong inductive biases—such as locality and translation equivariance—which enable efficient learning from relatively small datasets. Architectures like MobileNet and EfficientNet optimize for speed and efficiency, while deeper models like ResNet-50 consistently achieve high accuracy, reaching ~95% on benchmarks like CIFAR-10.

Vision Transformers (ViTs), by contrast, represent a fundamentally different approach. Rather than using convolutions, they treat images as sequences of patches and apply self-attention to model long-range dependencies across the entire image. Originally introduced in "Attention Is All You Need", the attention mechanism enables ViTs to learn more flexible, global representations—but often at the cost of increased data and compute requirements.

I chose to implement a ViT to explore how attention-based architectures can be applied to domains beyond natural language processing—specifically, computer vision. I was curious to see how shifting from convolutional priors to a fully attention-based model would affect the trade-offs around scalability, data efficiency, and the richness of learned representations.

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
