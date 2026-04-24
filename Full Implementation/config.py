"""
PruneVision AI - Central Configuration
All hyperparameters, paths, and settings for the self-pruning pipeline.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
EXPORT_DIR = os.path.join(OUTPUT_DIR, "exports")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# Create output directories
for d in [OUTPUT_DIR, CHECKPOINT_DIR, EXPORT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset ─────────────────────────────────────────────────────────────────
NUM_CLASSES = 10  # CIFAR-10 has 10 classes
IMAGE_SIZE = 224  # Standard for pretrained models (CIFAR-10 images resized from 32x32)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ─── Data Augmentation ──────────────────────────────────────────────────────
AUGMENTATION = {
    "random_resized_crop": {"size": IMAGE_SIZE, "scale": (0.8, 1.0)},
    "horizontal_flip_prob": 0.5,
    "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
    "rotation_degrees": 15,
}

# ImageNet normalization (for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─── Model ───────────────────────────────────────────────────────────────────
# Available: "mobilenetv3_small", "resnet18", "efficientnet_b0"
DEFAULT_MODEL = "hybrid"
PRETRAINED = True
FREEZE_EARLY_LAYERS = True  # Freeze early backbone layers for transfer learning

# ─── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE = 16
NUM_WORKERS = 0  # 0 for Windows compatibility
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30

# Optimizer
OPTIMIZER = "adam"  # "adam" or "sgd"
SGD_MOMENTUM = 0.9

# LR Scheduler
LR_SCHEDULER = "cosine"  # "cosine", "step", or "none"
LR_STEP_SIZE = 10
LR_GAMMA = 0.1

# ─── Self-Pruning / Gating ──────────────────────────────────────────────────
# Gate initialization bias (positive = gates start open)
GATE_INIT_BIAS = 3.0

# Pruning threshold: gates with sigmoid(g) < threshold are pruned
PRUNING_THRESHOLD = 0.05

# 3-Stage Sparsity Schedule (from PRD)
SPARSITY_SCHEDULE = {
    "stage1": {
        "name": "warm-up",
        "start_epoch": 0,
        "end_epoch": 10,
        "lambda_start": 0.0001,
        "lambda_end": 0.0001,
    },
    "stage2": {
        "name": "progressive",
        "start_epoch": 10,
        "end_epoch": 25,
        "lambda_start": 0.001,
        "lambda_end": 0.01,
    },
    "stage3": {
        "name": "fine-tuning",
        "start_epoch": 25,
        "end_epoch": 30,
        "lambda_start": 0.0,
        "lambda_end": 0.0,
    },
}

# ─── Deployment ──────────────────────────────────────────────────────────────
ONNX_OPSET_VERSION = 17
ONNX_DYNAMIC_AXES = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

# ─── Class Names ─────────────────────────────────────────────────────────────
# CIFAR-10 classes
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
