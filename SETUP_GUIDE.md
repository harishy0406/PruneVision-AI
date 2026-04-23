# 📚 PruneVision AI - Complete Setup & Usage Guide

## 📖 Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Deployment](#deployment)
6. [Dashboard](#dashboard)
7. [Docker Setup](#docker-setup)
8. [Troubleshooting](#troubleshooting)

---

## 🔧 Installation

### System Requirements

- **Python**: 3.9 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB for models and outputs
- **OS**: Windows, macOS, or Linux

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/PruneVision-AI.git
cd PruneVision-AI
```

### Step 2: Create Virtual Environment

```bash
# Using venv (recommended)
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
```

---

## ⚙️ Configuration

### Main Configuration File: `config.py`

```python
# Dataset Settings
NUM_CLASSES = 25
IMAGE_SIZE = 224
BATCH_SIZE = 16

# Training Settings
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Model Selection
DEFAULT_MODEL = "mobilenetv3_small"  # Options: "mobilenetv3_small", "resnet18", "efficientnet_b0"
PRETRAINED = True

# Pruning Settings
GATE_INIT_BIAS = 3.0
PRUNING_THRESHOLD = 0.05

# 3-Stage Sparsity Schedule
SPARSITY_SCHEDULE = {
    "stage1": {"start_epoch": 0, "end_epoch": 10, "lambda_start": 0.0001, "lambda_end": 0.0001},
    "stage2": {"start_epoch": 10, "end_epoch": 25, "lambda_start": 0.001, "lambda_end": 0.01},
    "stage3": {"start_epoch": 25, "end_epoch": 30, "lambda_start": 0.0, "lambda_end": 0.0},
}
```

### Quick Configuration Templates

#### For CPU Training (Recommended for Development)

```python
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 5e-4
DEFAULT_MODEL = "mobilenetv3_small"
PRETRAINED = True
```

#### For Production Training

```python
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
DEFAULT_MODEL = "mobilenetv3_small"
PRETRAINED = True
```

---

## 🎓 Training

### Basic Training

```bash
# Train MobileNetV3-Small
python train_model.py --model mobilenetv3_small --epochs 30

# Train ResNet-18
python train_model.py --model resnet18 --epochs 50

# Train EfficientNet-B0
python train_model.py --model efficientnet_b0 --epochs 40
```

### Advanced Training Options

```bash
# Custom learning rate
python train_model.py --model mobilenetv3_small --lr 0.0005

# Custom batch size
python train_model.py --model resnet18 --batch-size 32

# Custom pruning threshold
python train_model.py --model mobilenetv3_small --prune-threshold 0.1

# Without pretrained weights
python train_model.py --model mobilenetv3_small --no-pretrained

# Without freezing backbone
python train_model.py --model resnet18 --no-freeze

# All together
python train_model.py \
    --model mobilenetv3_small \
    --epochs 40 \
    --lr 0.001 \
    --batch-size 16 \
    --gate-bias 2.5 \
    --prune-threshold 0.05
```

### Training Output

Each training creates:

```
outputs/
├── checkpoints/
│   └── mobilenetv3_small/
│       ├── best_model.pth          # Best checkpoint
│       ├── final_model.pth         # Final trained model
│       ├── pruned_model.pth        # After hard pruning
│       ├── checkpoint_epoch_*.pth  # Per-epoch checkpoints
│       ├── training_history.json   # Loss/accuracy curves
│       └── evaluation_results.json # Final metrics
├── exports/
│   └── mobilenetv3_small_pruned.onnx  # ONNX export
└── logs/
    └── training_*.log              # Training logs
```

---

## 📊 Evaluation

### Automatic Evaluation

Evaluation runs automatically after training. Results saved in:

```
outputs/checkpoints/{model_name}/evaluation_results.json
```

### Manual Evaluation

```python
from prunevision.models import PrunedMobileNetV3
from prunevision.train.metrics import compute_metrics
import torch

# Load model
model = PrunedMobileNetV3(num_classes=25)
model.load_state_dict(torch.load("outputs/checkpoints/mobilenetv3_small/best_model.pth"))

# Get sparsity statistics
sparsity = model.get_sparsity_stats()
print(f"Global Sparsity: {sparsity['global_sparsity']:.2%}")
print(f"Model Size: {model.get_model_size_mb():.2f} MB")
```

### Performance Analysis

```bash
# View training history
python -c "import json; h = json.load(open('outputs/checkpoints/mobilenetv3_small/training_history.json')); print(f'Final Accuracy: {h[\"accuracy\"][-1]:.4f}')"

# View evaluation results
python -c "import json; r = json.load(open('outputs/checkpoints/mobilenetv3_small/evaluation_results.json')); print(json.dumps(r, indent=2))"
```

---

## 🚀 Deployment

### ONNX Export

Automatic (with `--export-onnx`):
```bash
python train_model.py --model mobilenetv3_small --export-onnx
```

Manual export:
```python
from prunevision.deploy.export_onnx import export_to_onnx

export_to_onnx(
    model,
    input_shape=(1, 3, 224, 224),
    output_path="model.onnx"
)
```

### Hard Pruning

```python
from prunevision.deploy.pruner import HardPruner

pruner = HardPruner(model, threshold=0.05)
pruned_model = pruner.prune()
```

### Model Serving

Using FastAPI (optional):
```bash
python -m uvicorn api:app --port 8000
```

---

## 📊 Dashboard

### Launch Dashboard

```bash
streamlit run app_advanced.py
```

Access at: `http://localhost:8501`

### Dashboard Features

1. **📋 Overview**
   - Project information
   - Key metrics summary
   - Pipeline visualization

2. **📊 Dataset Explorer**
   - Class distribution charts
   - Sample image browser
   - Dataset statistics

3. **⚙️ Training Monitor**
   - Loss curves
   - Accuracy progression
   - Sparsity evolution
   - λ schedule visualization

4. **🔍 Model Analysis**
   - Layer-wise sparsity
   - Gate value distributions
   - Parameter reduction breakdown
   - Model comparison

5. **📤 Export & Deployment**
   - ONNX export commands
   - Quantization options
   - Model comparison table
   - Size estimation

6. **🎯 Live Inference**
   - Upload image for classification
   - Real-time predictions
   - Confidence scores
   - Processing time

---

## 🐳 Docker Setup

### Build Image

```bash
docker build -t prunevision-ai:latest .
```

### Run Container

```bash
# Simple run
docker run -p 8501:8501 prunevision-ai:latest

# With volume for outputs
docker run -p 8501:8501 -v $(pwd)/outputs:/app/outputs prunevision-ai:latest

# With data volume
docker run -p 8501:8501 \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/data:/app/data:ro \
    prunevision-ai:latest
```

### Docker Compose

```bash
# Development setup (just app)
docker-compose up app

# Full stack with monitoring
docker-compose --profile with-db --profile with-cache --profile with-monitoring up

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### Docker Compose Profiles

- **Default**: App only
- **with-api**: Add FastAPI server
- **with-db**: Add PostgreSQL database
- **with-cache**: Add Redis cache
- **with-monitoring**: Add Prometheus & Grafana
- **with-proxy**: Add Nginx reverse proxy

---

## 🔧 Troubleshooting

### Issue: PyTorch Installation Failed

**Solution**:
```bash
# Install CPU version specifically
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue: CUDA Not Found (GPU)

**Solution**: We support CPU-only. Use:
```bash
# Ensure CPU-only installation
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Memory Error During Training

**Solution**: Reduce batch size in `config.py`:
```python
BATCH_SIZE = 8  # Reduce from 16
```

### Issue: Streamlit Port Already in Use

**Solution**: Use different port:
```bash
streamlit run app_advanced.py --server.port 8502
```

### Issue: Data Not Found

**Solution**: Ensure data structure:
```
data/
└── images/
    ├── BEANS/
    ├── CAKE/
    └── ... (25 classes)
```

### Issue: Model Download Fails

**Solution**: Download manually:
```bash
python -c "from torchvision.models import mobilenet_v3_small; mobilenet_v3_small(pretrained=True)"
```

### Issue: ONNX Export Fails

**Solution**: Update onnx dependencies:
```bash
pip install --upgrade onnx onnxruntime
```

---

## 📝 Example Workflows

### Workflow 1: Quick Test (5 minutes)

```bash
# 1. Train minimal model
python train_model.py --model mobilenetv3_small --epochs 5

# 2. View results
python -c "import json; print(json.load(open('outputs/checkpoints/mobilenetv3_small/training_history.json'))['accuracy'][-1])"

# 3. Launch dashboard
streamlit run app_advanced.py
```

### Workflow 2: Production Training (30 minutes)

```bash
# 1. Train with all options
python train_model.py \
    --model mobilenetv3_small \
    --epochs 30 \
    --batch-size 16 \
    --lr 0.001 \
    --export-onnx

# 2. Analyze results
# (Open dashboard and review)

# 3. Deploy
docker build -t prunevision-ai:v1.0.0 .
docker run -p 8501:8501 prunevision-ai:v1.0.0
```

### Workflow 3: Model Comparison

```bash
# Train all models
for model in mobilenetv3_small resnet18 efficientnet_b0; do
    python train_model.py --model $model --epochs 30 --export-onnx
done

# Compare in dashboard
streamlit run app_advanced.py
```

---

## 📚 Additional Resources

- [README.md](README.md) - Project overview
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Cloud deployment
- [GIT_COMMIT_STRATEGY.md](GIT_COMMIT_STRATEGY.md) - Development workflow
- [GITHUB_COPILOT_PROMPTS.md](Claud/GITHUB_COPILOT_PROMPTS.md) - AI assistance prompts

---

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/PruneVision-AI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/PruneVision-AI/discussions)
- **Email**: support@prunevision-ai.com
