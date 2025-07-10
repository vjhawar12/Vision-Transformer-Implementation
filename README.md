# Vision Transformer (ViT) - From Scratch

## 🔍 Overview

This project is a PyTorch-based implementation of the research paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". It builds the Vision Transformer (ViT) architecture from scratch, including core components such as patch embeddings, multi-head self-attention, and transformer encoders. The training pipeline incorporates regularization techniques like stochastic depth and dropout, along with data augmentation methods such as CutMix, MixUp, horizontal flipping, and color jittering. The model is trained and evaluated on the CIFAR-10 dataset, achieving a final test accuracy of 85.7%.

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

1. Open the notebook:

```bash
jupyter notebook Vision_Transformer_from_scratch.ipynb
```

Or use Google Colab.

2. Run all cells sequentially.
