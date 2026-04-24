#!/usr/bin/env python
"""
Quick verification script for PruneVision AI setup.
"""

import sys
import torch
sys.path.insert(0, '.')

print("=" * 60)
print("PruneVision AI - Project Verification")
print("=" * 60)

# Test 1: Config
try:
    import config
    print(f"✓ Config loaded: {config.NUM_CLASSES} classes, {config.DEFAULT_MODEL}")
except Exception as e:
    print(f"✗ Config failed: {e}")
    sys.exit(1)

# Test 2: PyTorch
try:
    print(f"✓ PyTorch {torch.__version__}")
except Exception as e:
    print(f"✗ PyTorch failed: {e}")
    sys.exit(1)

# Test 3: Gates Module
try:
    from prunevision.gates import GateLayer
    gate = GateLayer(64, init_bias=3.0)
    print(f"✓ Gates module: GateLayer instantiated")
except Exception as e:
    print(f"✗ Gates module failed: {e}")
    sys.exit(1)

# Test 4: Models
try:
    from prunevision.models import PrunedMobileNetV3, PrunedResNet18, PrunedEfficientNetB0
    model = PrunedMobileNetV3(num_classes=25, pretrained=False)
    total_params = model.get_total_params()
    size_mb = model.get_model_size_mb()
    print(f"✓ Models module: MobileNetV3 ({total_params:,} params, {size_mb:.2f} MB)")
except Exception as e:
    print(f"✗ Models failed: {e}")
    sys.exit(1)

# Test 5: Forward Pass
try:
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        output = model(x)
    assert output.shape == (2, 25), f"Expected (2, 25), got {output.shape}"
    print(f"✓ Forward pass: Input {x.shape} → Output {output.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    sys.exit(1)

# Test 6: Sparsity
try:
    sparsity = model.get_sparsity_stats()
    print(f"✓ Sparsity analysis: {sparsity['global_sparsity']:.2%} global sparsity")
except Exception as e:
    print(f"✗ Sparsity failed: {e}")
    sys.exit(1)

# Test 7: Data Module
try:
    from prunevision.data.dataset import RetailDataset
    dataset = RetailDataset([], [], class_names=config.CLASS_NAMES)
    print(f"✓ Data module: RetailDataset initialized")
except Exception as e:
    print(f"✗ Data module failed: {e}")
    sys.exit(1)

# Test 8: Training Module
try:
    from prunevision.train.scheduler import SparsityScheduler
    scheduler = SparsityScheduler(schedule=config.SPARSITY_SCHEDULE, total_epochs=config.EPOCHS)
    lambda_val = scheduler.get_lambda(epoch=15)
    print(f"✓ Training module: SparsityScheduler (λ[epoch 15] = {lambda_val:.6f})")
except Exception as e:
    print(f"✗ Training module failed: {e}")
    sys.exit(1)

# Test 9: Deployment Module
try:
    from prunevision.deploy.pruner import HardPruner
    pruner = HardPruner(threshold=0.05)
    print(f"✓ Deployment module: HardPruner initialized")
except Exception as e:
    print(f"✗ Deployment module failed: {e}")
    sys.exit(1)

# Test 10: Analysis Module
try:
    from prunevision.analysis.analyzer import SparsityAnalyzer
    analyzer = SparsityAnalyzer(model)
    print(f"✓ Analysis module: SparsityAnalyzer initialized")
except Exception as e:
    print(f"✗ Analysis module failed: {e}")
    sys.exit(1)

print("=" * 60)
print("✓ All tests passed! Project is ready to use.")
print("=" * 60)
print("\nNext steps:")
print("1. Train a model: python train_model.py --model mobilenetv3_small")
print("2. Launch dashboard: streamlit run app_advanced.py")
print("3. See SETUP_GUIDE.md for detailed instructions")
