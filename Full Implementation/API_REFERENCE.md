# 🔌 PruneVision AI - API Reference

## Core APIs

### Gate Mechanism (`prunevision.gates`)

#### `GateLayer(num_gates, init_bias=3.0)`

Learnable gating mechanism for neural network pruning.

**Parameters:**
- `num_gates` (int): Number of gate parameters
- `init_bias` (float): Initial gate bias (positive = open gates)

**Methods:**
- `forward(x)`: Apply gates to tensor
- `get_sparsity(threshold)`: Compute sparsity ratio
- `get_gate_values()`: Get sigmoid(gate_params)

**Example:**
```python
from prunevision.gates import GateLayer
import torch

gate = GateLayer(num_gates=64, init_bias=3.0)
x = torch.randn(2, 64, 32, 32)
output = gate(x)  # Shape: (2, 64, 32, 32)
sparsity = gate.get_sparsity(threshold=0.05)
```

---

### Models (`prunevision.models`)

#### `PrunedMobileNetV3(num_classes=25, pretrained=True)`

MobileNetV3-Small with gating layers.

**Parameters:**
- `num_classes` (int): Number of output classes
- `pretrained` (bool): Load ImageNet weights

**Methods:**
- `forward(x)`: Forward pass
- `get_total_params()`: Total parameters
- `get_sparsity_stats(threshold=0.05)`: Sparsity statistics
- `get_model_size_mb()`: Model size in MB
- `freeze_backbone(freeze=True)`: Freeze backbone layers

**Example:**
```python
from prunevision.models import PrunedMobileNetV3
import torch

model = PrunedMobileNetV3(num_classes=25, pretrained=True)
x = torch.randn(2, 3, 224, 224)
logits = model(x)  # Shape: (2, 25)
```

---

#### `PrunedResNet18(num_classes=25, pretrained=True)`

ResNet-18 with gating layers.

---

#### `PrunedEfficientNetB0(num_classes=25, pretrained=True)`

EfficientNet-B0 with gating layers.

---

### Training (`prunevision.train`)

#### `PruneVisionTrainer(model, train_loader, val_loader, ...)`

Main training engine with integrated self-pruning.

**Parameters:**
- `model`: PrunableModel instance
- `train_loader`: Training DataLoader
- `val_loader`: Validation DataLoader
- `class_weights`: Tensor of per-class weights
- `device`: Device ("cpu" or "cuda")

**Methods:**
- `train(epochs)`: Train for specified epochs
- `validate()`: Validate on validation set
- `save_checkpoint(path, best=False)`: Save model

**Example:**
```python
from prunevision.train import PruneVisionTrainer
from prunevision.models import PrunedMobileNetV3

model = PrunedMobileNetV3(num_classes=25)
trainer = PruneVisionTrainer(model, train_loader, val_loader)
trainer.train(epochs=30)
```

---

#### `SparsityScheduler(base_lambda, schedule_config)`

3-stage sparsity scheduling for pruning.

**Example:**
```python
from prunevision.train import SparsityScheduler
import config

scheduler = SparsityScheduler(base_lambda=0.001, schedule_config=config.SPARSITY_SCHEDULE)
lambda_val = scheduler.get_lambda(epoch=15)  # Get λ for epoch 15
```

---

### Data (`prunevision.data`)

#### `RetailDataset(image_paths, labels, transform=None)`

Custom dataset for retail product images.

**Methods:**
- `__getitem__(idx)`: Get image and label
- `__len__()`: Dataset size
- `get_class_distribution()`: Class counts
- `get_class_weights()`: Inverse-frequency weights

**Example:**
```python
from prunevision.data import RetailDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

dataset = RetailDataset(image_paths, labels, transform=transform)
```

---

#### `get_dataloaders(data_dir, batch_size=16, ...)`

Create train/val/test dataloaders.

**Example:**
```python
from prunevision.data import get_dataloaders

train_loader, val_loader, test_loader, class_weights, class_names = get_dataloaders(
    data_dir="data/images",
    batch_size=16
)
```

---

### Deployment (`prunevision.deploy`)

#### `HardPruner(model, threshold=0.05)`

Permanently apply pruning to model.

**Methods:**
- `prune()`: Apply hard pruning
- `save(path)`: Save pruned model

**Example:**
```python
from prunevision.deploy import HardPruner

pruner = HardPruner(model, threshold=0.05)
pruned_model = pruner.prune()
pruner.save("pruned_model.pth")
```

---

#### `export_to_onnx(model, input_shape, output_path)`

Export model to ONNX format.

**Example:**
```python
from prunevision.deploy import export_to_onnx

export_to_onnx(
    model,
    input_shape=(1, 3, 224, 224),
    output_path="model.onnx"
)
```

---

### Analysis (`prunevision.analysis`)

#### `SparsityAnalyzer(model)`

Analyze model sparsity and layer importance.

**Methods:**
- `get_layer_sparsity()`: Per-layer sparsity
- `get_gate_statistics()`: Gate value statistics
- `get_pruning_candidates()`: Layers suitable for pruning

**Example:**
```python
from prunevision.analysis import SparsityAnalyzer

analyzer = SparsityAnalyzer(model)
layer_sparsity = analyzer.get_layer_sparsity()
candidates = analyzer.get_pruning_candidates(threshold=0.1)
```

---

## Configuration API

### `config.py`

Central configuration file.

**Key Settings:**
```python
# Dataset
NUM_CLASSES = 25
IMAGE_SIZE = 224
BATCH_SIZE = 16

# Training
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Model
DEFAULT_MODEL = "mobilenetv3_small"
PRETRAINED = True

# Pruning
GATE_INIT_BIAS = 3.0
PRUNING_THRESHOLD = 0.05
SPARSITY_SCHEDULE = {...}
```

---

## CLI API

### Training Script (`train_model.py`)

```bash
python train_model.py [options]

Options:
  --model {mobilenetv3_small, resnet18, efficientnet_b0}
  --epochs EPOCHS
  --lr LEARNING_RATE
  --batch-size BATCH_SIZE
  --no-pretrained
  --no-freeze
  --gate-bias BIAS
  --prune-threshold THRESHOLD
  --export-onnx
```

---

## Streamlit Dashboard API

### App Entry Point (`app_advanced.py`)

```bash
streamlit run app_advanced.py
```

**Available Pages:**
1. Overview
2. Dataset Explorer
3. Training Monitor
4. Model Analysis
5. Export & Deployment
6. Live Inference

---

## Example: End-to-End Workflow

```python
import torch
from prunevision.models import PrunedMobileNetV3
from prunevision.train import PruneVisionTrainer
from prunevision.data import get_dataloaders
from prunevision.deploy import HardPruner, export_to_onnx
import config

# 1. Load data
train_loader, val_loader, test_loader, class_weights, _ = get_dataloaders(
    data_dir=config.DATA_DIR,
    batch_size=config.BATCH_SIZE
)

# 2. Create model
model = PrunedMobileNetV3(
    num_classes=config.NUM_CLASSES,
    pretrained=config.PRETRAINED
)

# 3. Train with pruning
trainer = PruneVisionTrainer(
    model,
    train_loader,
    val_loader,
    class_weights=class_weights,
    device="cpu"
)
trainer.train(epochs=config.EPOCHS)

# 4. Hard prune
pruner = HardPruner(model, threshold=config.PRUNING_THRESHOLD)
pruned_model = pruner.prune()

# 5. Export to ONNX
export_to_onnx(
    pruned_model,
    input_shape=(1, 3, 224, 224),
    output_path="pruned_model.onnx"
)

# 6. Evaluate
val_loss, val_acc = trainer.validate()
print(f"Final Accuracy: {val_acc:.4f}")
```

---

## Version Information

- **PyTorch**: 2.0+
- **Python**: 3.9+
- **ONNX**: 1.14+

---

## Support

For API questions and issues, see:
- GitHub Issues: https://github.com/yourusername/PruneVision-AI/issues
- Documentation: [README.md](README.md)
