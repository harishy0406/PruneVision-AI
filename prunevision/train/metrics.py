"""
PruneVision AI - Training Metrics
Functions for computing accuracy, sparsity, and inference latency metrics.
"""

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Optional class name list for report.
        
    Returns:
        Dictionary with accuracy, precision, recall, F1, per-class metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    # Per-class metrics
    per_class_p, per_class_r, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
    )
    
    result = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": {},
    }
    
    num_classes = len(per_class_p)
    for i in range(num_classes):
        name = class_names[i] if class_names and i < len(class_names) else str(i)
        result["per_class"][name] = {
            "precision": per_class_p[i],
            "recall": per_class_r[i],
            "f1": per_class_f1[i],
            "support": int(per_class_support[i]),
        }
    
    return result


def compute_topk_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """
    Compute top-k accuracy from model outputs.
    
    Args:
        outputs: Model logits (B, C).
        targets: Ground truth labels (B,).
        topk: Tuple of k values.
        
    Returns:
        Dictionary with top-k accuracy values.
    """
    maxk = max(topk)
    batch_size = targets.size(0)
    
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    
    result = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        result[f"top{k}"] = (correct_k / batch_size).item()
    
    return result


def measure_inference_latency(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    num_warmup: int = 5,
    num_runs: int = 20,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Measure model inference latency.
    
    Args:
        model: PyTorch model.
        input_size: Input tensor shape.
        num_warmup: Number of warmup runs.
        num_runs: Number of timed runs.
        device: Device to run on.
        
    Returns:
        Dictionary with mean, std, min, max latency in milliseconds.
    """
    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "median_ms": np.median(latencies),
    }


def get_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def get_classification_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
) -> str:
    """Generate formatted classification report."""
    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )
