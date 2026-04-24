"""
PruneVision AI - Sparsity Analyzer
Tools for analyzing model sparsity, gate distributions, layer importance,
and generating visualizations.
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

from ..gates.gate_layer import GateLayer
from ..gates.gate_wrapper import collect_gate_layers, compute_model_sparsity


class SparsityAnalyzer:
    """
    Comprehensive analysis tool for self-pruning models.
    
    Provides:
    - Layer-wise sparsity analysis
    - Gate value distributions
    - Parameter reduction metrics
    - Accuracy vs sparsity trade-off analysis
    - Layer importance ranking
    
    Args:
        model: PrunableModel to analyze.
        threshold: Gate threshold for sparsity computation.
    """
    
    def __init__(self, model: nn.Module, threshold: float = None):
        self.model = model
        self.threshold = threshold or config.PRUNING_THRESHOLD
    
    def get_layer_sparsity(self) -> List[Dict]:
        """
        Get per-layer sparsity statistics.
        
        Returns:
            List of dicts with layer name, num_gates, active, pruned, sparsity.
        """
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, GateLayer):
                gate_values = module.get_gate_values().cpu().numpy()
                layers.append({
                    "name": name,
                    "num_gates": module.num_gates,
                    "active": int((gate_values >= self.threshold).sum()),
                    "pruned": int((gate_values < self.threshold).sum()),
                    "sparsity": float(module.get_sparsity(self.threshold)),
                    "mean_gate": float(gate_values.mean()),
                    "min_gate": float(gate_values.min()),
                    "max_gate": float(gate_values.max()),
                    "std_gate": float(gate_values.std()),
                })
        return layers
    
    def get_gate_distributions(self) -> Dict[str, np.ndarray]:
        """
        Get gate value distributions for each gated layer.
        
        Returns:
            Dict mapping layer name to array of gate values.
        """
        distributions = {}
        for name, module in self.model.named_modules():
            if isinstance(module, GateLayer):
                distributions[name] = module.get_gate_values().cpu().numpy()
        return distributions
    
    def get_global_summary(self) -> Dict:
        """
        Get comprehensive global model summary.
        
        Returns:
            Dict with overall model statistics.
        """
        sparsity_stats = compute_model_sparsity(self.model, self.threshold)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        gate_params = sum(
            m.gate_params.numel() 
            for m in self.model.modules() 
            if isinstance(m, GateLayer)
        )
        
        # Count non-zero weight parameters (excluding gates)
        nonzero_weight_params = 0
        total_weight_params = 0
        for name, param in self.model.named_parameters():
            if "gate_params" not in name:
                total_weight_params += param.numel()
                nonzero_weight_params += (param != 0).sum().item()
        
        return {
            "total_parameters": total_params,
            "gate_parameters": gate_params,
            "weight_parameters": total_weight_params,
            "nonzero_weight_parameters": nonzero_weight_params,
            "global_sparsity": sparsity_stats["global_sparsity"],
            "total_gates": sparsity_stats["total_gates"],
            "active_gates": sparsity_stats["total_active"],
            "pruned_gates": sparsity_stats["total_pruned"],
            "model_size_mb": (total_params * 4) / (1024 * 1024),
            "effective_size_mb": (nonzero_weight_params * 4) / (1024 * 1024),
            "compression_ratio": total_weight_params / max(nonzero_weight_params, 1),
        }
    
    def get_layer_importance_ranking(self) -> List[Dict]:
        """
        Rank layers by importance (mean gate value).
        Higher mean gate value = more important layer.
        
        Returns:
            List of layers sorted by importance (descending).
        """
        layers = self.get_layer_sparsity()
        return sorted(layers, key=lambda x: x["mean_gate"], reverse=True)
    
    def get_pruning_candidates(self, max_sparsity: float = 0.9) -> List[Dict]:
        """
        Identify layers that could be further pruned.
        
        Args:
            max_sparsity: Maximum sparsity threshold for candidate layers.
            
        Returns:
            List of layers with sparsity below max_sparsity.
        """
        layers = self.get_layer_sparsity()
        return [l for l in layers if l["sparsity"] < max_sparsity]
    
    def compare_models(
        self,
        baseline_params: int,
        baseline_size_mb: float,
        baseline_accuracy: float,
        pruned_accuracy: float,
    ) -> Dict:
        """
        Compare pruned model against baseline metrics.
        
        Args:
            baseline_params: Total params in baseline model.
            baseline_size_mb: Baseline model size in MB.
            baseline_accuracy: Baseline model accuracy.
            pruned_accuracy: Pruned model accuracy.
            
        Returns:
            Comparison report dict.
        """
        summary = self.get_global_summary()
        
        return {
            "baseline": {
                "params": baseline_params,
                "size_mb": baseline_size_mb,
                "accuracy": baseline_accuracy,
            },
            "pruned": {
                "params": summary["nonzero_weight_parameters"],
                "size_mb": summary["effective_size_mb"],
                "accuracy": pruned_accuracy,
            },
            "reduction": {
                "param_reduction": 1.0 - (summary["nonzero_weight_parameters"] / baseline_params),
                "size_reduction": 1.0 - (summary["effective_size_mb"] / baseline_size_mb),
                "accuracy_drop": baseline_accuracy - pruned_accuracy,
            },
            "global_sparsity": summary["global_sparsity"],
        }
    
    def generate_report(self) -> str:
        """Generate a formatted text report of model analysis."""
        summary = self.get_global_summary()
        layers = self.get_layer_sparsity()
        
        lines = [
            "=" * 70,
            "PruneVision AI - Model Analysis Report",
            "=" * 70,
            "",
            "Global Statistics:",
            f"  Total Parameters:     {summary['total_parameters']:>12,}",
            f"  Gate Parameters:      {summary['gate_parameters']:>12,}",
            f"  Weight Parameters:    {summary['weight_parameters']:>12,}",
            f"  Non-zero Weights:     {summary['nonzero_weight_parameters']:>12,}",
            f"  Global Sparsity:      {summary['global_sparsity']:>11.1%}",
            f"  Model Size:           {summary['model_size_mb']:>10.2f} MB",
            f"  Effective Size:       {summary['effective_size_mb']:>10.2f} MB",
            f"  Compression Ratio:    {summary['compression_ratio']:>10.2f}x",
            "",
            "Layer-wise Sparsity:",
            f"  {'Layer':<40s} {'Gates':>6s} {'Active':>7s} {'Pruned':>7s} {'Sparsity':>9s} {'Mean':>6s}",
            "-" * 80,
        ]
        
        for layer in layers:
            lines.append(
                f"  {layer['name']:<40s} "
                f"{layer['num_gates']:>6d} "
                f"{layer['active']:>7d} "
                f"{layer['pruned']:>7d} "
                f"{layer['sparsity']:>8.1%} "
                f"{layer['mean_gate']:>6.3f}"
            )
        
        lines.append("=" * 70)
        return "\n".join(lines)
