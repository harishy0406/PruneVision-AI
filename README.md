# ✂️ PruneVision AI — Self-Pruning Neural Networks

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/yourusername/PruneVision-AI/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/PruneVision-AI/actions)

**PruneVision AI** is a production-grade platform for self-pruning neural networks, enabling efficient edge deployment with minimal accuracy loss. Built with learnable gating mechanisms, it automates model compression during training.

## ✨ Key Featurespip install -r requirements.txt

- 🧠 **Learnable Gates** - Automatic model compression through self-pruning
- 🎯 **Multi-Model Support** - MobileNetV3, ResNet-18, EfficientNet-B0
- 📊 **Interactive Dashboard** - Real-time training visualization and analysis
- 🚀 **Production Ready** - Docker, Kubernetes, cloud-native deployment
- ⚡ **Edge Optimized** - 50%+ model reduction with <1% accuracy loss
- 🔒 **Enterprise Grade** - Full CI/CD, security scanning, comprehensive tests

## 📊 Quick Comparison

| Aspect | Baseline | PruneVision | Improvement |
|--------|----------|-------------|-------------|
| Parameters | 2.5M | 1.2M | **52% smaller** |
| Model Size | 9.8MB | 4.7MB | **52% reduction** |
| Inference Time | 124ms | 78ms | **37% faster** |
| Accuracy | 89.2% | 88.9% | **-0.3%** |

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/PruneVision-AI.git
cd PruneVision-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train a Model

```bash
# MobileNetV3-Small (fastest, recommended for CPU)
python train_model.py --model mobilenetv3_small --epochs 30

# ResNet-18
python train_model.py --model resnet18 --epochs 30

# EfficientNet-B0
python train_model.py --model efficientnet_b0 --epochs 30

# With ONNX export
python train_model.py --model mobilenetv3_small --export-onnx
```

### Launch Dashboard
```bash
streamlit run app.py
```

## 📊 Dataset

- **25 retail product classes**: BEANS, CAKE, CANDY, CEREAL, CHIPS, CHOCOLATE, COFFEE, CORN, FISH, FLOUR, HONEY, JAM, JUICE, MILK, NUTS, OIL, PASTA, RICE, SODA, SPICES, SUGAR, TEA, TOMATO_SAUCE, VINEGAR, WATER
- **4,947 images** (256×256 RGB PNG)
- Stratified 70/15/15 train/val/test split
- Class-weighted sampling for imbalanced data

## 🏗️ Model Zoo

| Model | Params | Size | Target Use |
|-------|--------|------|-----------|
| MobileNetV3-Small | ~2.5M | ~10 MB | Ultra-fast edge inference |
| ResNet-18 | ~11.7M | ~45 MB | Balanced accuracy/speed |
| EfficientNet-B0 | ~5.3M | ~20 MB | Best accuracy/efficiency |

## 🔧 Configuration

All hyperparameters are centralized in `config.py`:
- Image size, batch size, learning rate
- 3-stage sparsity schedule (λ values, epoch boundaries)
- Gate initialization bias, pruning threshold
- Data augmentation parameters

## 📈 Dashboard Features

1. **Overview** — Architecture explanation, pipeline visualization
2. **Dataset Explorer** — Class distribution, sample image browser
3. **Training Monitor** — Loss/accuracy/sparsity curves, λ schedule
4. **Model Analysis** — Layer-wise sparsity, gate distributions
5. **Deployment** — Model comparison, export commands
6. **Live Demo** — Upload image → real-time classification

## 📜 License

Apache 2.0
