"""
PruneVision AI - Training Script
CLI interface for training self-pruning models on retail product datasets.

Usage:
    python train_model.py --model mobilenetv3_small --epochs 30
    python train_model.py --model resnet18 --epochs 50 --lr 0.0005
    python train_model.py --model efficientnet_b0 --no-freeze
"""

import argparse
import os
import json
import time

import torch

import config
from prunevision.data.dataset import get_dataloaders
from prunevision.models import PrunedMobileNetV3, PrunedResNet18, PrunedEfficientNetB0, PrunedHybrid
from prunevision.train.trainer import PruneVisionTrainer
from prunevision.train.metrics import measure_inference_latency, get_classification_report
from prunevision.deploy.pruner import HardPruner
from prunevision.analysis.analyzer import SparsityAnalyzer


MODEL_REGISTRY = {
    "mobilenetv3_small": PrunedMobileNetV3,
    "resnet18": PrunedResNet18,
    "efficientnet_b0": PrunedEfficientNetB0,
    "hybrid": PrunedHybrid,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="PruneVision AI - Train Self-Pruning Neural Networks"
    )
    parser.add_argument(
        "--model", type=str, default=config.DEFAULT_MODEL,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture to train",
    )
    parser.add_argument(
        "--epochs", type=int, default=config.EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr", type=float, default=config.LEARNING_RATE,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size", type=int, default=config.BATCH_SIZE,
        help="Batch size",
    )
    parser.add_argument(
        "--no-pretrained", action="store_true",
        help="Don't use pretrained weights",
    )
    parser.add_argument(
        "--no-freeze", action="store_true",
        help="Don't freeze backbone layers",
    )
    parser.add_argument(
        "--gate-bias", type=float, default=config.GATE_INIT_BIAS,
        help="Initial gate bias",
    )
    parser.add_argument(
        "--prune-threshold", type=float, default=config.PRUNING_THRESHOLD,
        help="Pruning threshold for gates",
    )
    parser.add_argument(
        "--data-dir", type=str, default=config.DATA_DIR,
        help="Path to dataset",
    )
    parser.add_argument(
        "--output-dir", type=str, default=config.OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--export-onnx", action="store_true",
        help="Export to ONNX after training",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"  PruneVision AI - Self-Pruning Neural Network Training")
    print(f"{'='*60}")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # ─── Data ────────────────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    
    # ─── Model ───────────────────────────────────────────────────────────────
    print("\n[2/5] Building model...")
    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass(
        num_classes=config.NUM_CLASSES,
        pretrained=not args.no_pretrained,
        gate_init_bias=args.gate_bias,
        freeze_backbone=not args.no_freeze,
    )
    print(model.summary())
    
    # ─── Train ───────────────────────────────────────────────────────────────
    print("\n[3/5] Training...")
    save_dir = os.path.join(args.output_dir, "checkpoints", args.model)
    os.makedirs(save_dir, exist_ok=True)
    
    trainer = PruneVisionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        class_names=class_names,
        device=device,
        config_override={"epochs": args.epochs, "lr": args.lr},
    )
    
    start_time = time.time()
    history = trainer.train(save_dir=save_dir)
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # ─── Evaluate ────────────────────────────────────────────────────────────
    print("\n[4/5] Evaluating on test set...")
    eval_results = trainer.evaluate(test_loader)
    
    print(f"\nTest Results:")
    print(f"  Top-1 Accuracy: {eval_results['top1']:.2%}")
    print(f"  Top-5 Accuracy: {eval_results['top5']:.2%}")
    print(f"  F1 Score: {eval_results['f1']:.4f}")
    print(f"  Sparsity: {eval_results['sparsity']['global_sparsity']:.1%}")
    print(f"  Latency: {eval_results['latency']['mean_ms']:.1f} ms")
    print(f"  Model Size: {eval_results['model_size_mb']:.2f} MB")
    
    # Classification report
    print(f"\nClassification Report:")
    # Re-compute for the report
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())
    
    print(get_classification_report(all_labels, all_preds, class_names))
    
    # ─── Pruning & Analysis ──────────────────────────────────────────────────
    print("\n[5/5] Pruning & Analysis...")
    
    # Hard prune
    pruner = HardPruner(threshold=args.prune_threshold)
    report = pruner.get_pruning_report(model)
    
    print(f"\nPruning Report:")
    print(f"  Parameters Before: {report['total_params_before']:,}")
    print(f"  Non-zero After:    {report['nonzero_params_after']:,}")
    print(f"  Reduction:         {report['parameter_reduction']:.1%}")
    print(f"  Size Before:       {report['size_before_mb']:.2f} MB")
    print(f"  Size After:        {report['size_after_mb']:.2f} MB")
    
    # Sparsity Analysis
    analyzer = SparsityAnalyzer(model, threshold=args.prune_threshold)
    print(f"\n{analyzer.generate_report()}")
    
    # Save pruned model
    pruned_model = pruner.prune(model, remove_gates=True)
    pruned_path = os.path.join(save_dir, "pruned_model.pth")
    torch.save(pruned_model.state_dict(), pruned_path)
    print(f"\nPruned model saved to {pruned_path}")
    
    # Save evaluation results
    eval_path = os.path.join(save_dir, "evaluation_results.json")
    # Convert non-serializable values
    serializable_results = {
        "accuracy": eval_results["accuracy"],
        "top1": eval_results["top1"],
        "top5": eval_results["top5"],
        "f1": eval_results["f1"],
        "precision": eval_results["precision"],
        "recall": eval_results["recall"],
        "sparsity": eval_results["sparsity"]["global_sparsity"],
        "latency_ms": eval_results["latency"]["mean_ms"],
        "model_size_mb": eval_results["model_size_mb"],
        "total_params": eval_results["total_params"],
        "param_reduction": report["parameter_reduction"],
        "training_time_s": total_time,
    }
    with open(eval_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Evaluation results saved to {eval_path}")
    
    # ONNX Export
    if args.export_onnx:
        print("\nExporting to ONNX...")
        from prunevision.deploy.export_onnx import export_to_onnx, validate_onnx_model
        onnx_path = os.path.join(config.EXPORT_DIR, f"{args.model}_pruned.onnx")
        export_to_onnx(pruned_model, output_path=onnx_path)
        
        validation = validate_onnx_model(onnx_path, pruned_model)
        print(f"ONNX Validation: {'PASSED' if validation['valid'] else 'FAILED'}")
        if validation.get("file_size_mb"):
            print(f"ONNX File Size: {validation['file_size_mb']:.2f} MB")
    
    print(f"\n{'='*60}")
    print(f"  Training Complete! All outputs saved to: {save_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
