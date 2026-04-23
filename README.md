# ✂️ PruneVision AI

**Self-Pruning Neural Networks for Efficient Edge-Based Retail Image Classification**

A lightweight retail image classification system that uses self-pruning neural networks to dynamically remove redundant parameters during training, enabling efficient and fast inference on edge devices without significant accuracy loss.

---

## 🧠 How It Works

PruneVision AI integrates **learnable gating mechanisms** directly into the training pipeline:

1. Every convolution channel has a learnable **gate parameter** `g`
2. During forward pass: `masked_weight = weight × sigmoid(g)`
3. **L1 regularization** on gates drives unimportant channels toward zero
4. Post-training, gates below threshold are permanently pruned

### 3-Stage Training Schedule
| Stage | Epochs | Lambda (λ) | Purpose |
|-------|--------|-----------|---------|
| Warm-up | 1-10 | 0.0001 | Learn representations |
| Progressive | 11-25 | 0.001→0.01 | Drive pruning |
| Fine-tuning | 26-30 | 0.0 | Polish accuracy |

## 📁 Project Structure

```
PruneVision-AI/
├── prunevision/
│   ├── gates/          # Core gating mechanism (GateLayer, GatedConv2d)
│   ├── models/         # Model zoo (MobileNetV3, ResNet-18, EfficientNet-B0)
│   ├── train/          # Training engine, sparsity scheduler, metrics
│   ├── data/           # Dataset loading, augmentation, splits
│   ├── deploy/         # Hard pruning, ONNX export
│   └── analysis/       # Sparsity analysis, layer importance
├── app.py              # Streamlit dashboard
├── train_model.py      # CLI training script
├── config.py           # Central configuration
└── data/images/        # Retail product dataset (25 classes)
```

## 🚀 Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train a Model
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
