"""
PruneVision AI - Hard Pruner
Permanently applies gate masks to produce a clean, sparse model.
After hard pruning, gated-out weights are zeroed and gate parameters
can be removed, producing a standard PyTorch model.
"""

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

from ..gates.gate_layer import GateLayer
from ..gates.gate_wrapper import GatedConv2d, GatedLinear, compute_model_sparsity


class HardPruner:
    """
    Applies hard pruning to a model by permanently zeroing out
    weights corresponding to pruned gates.
    
    After hard pruning:
    - Weights with gate values below threshold are set to zero
    - The model becomes structurally sparse
    - Gate parameters can optionally be removed
    
    Args:
        threshold (float): Gate value threshold below which weights are pruned.
    """
    
    def __init__(self, threshold: float = None):
        self.threshold = threshold or config.PRUNING_THRESHOLD
    
    def prune(self, model: nn.Module, remove_gates: bool = False) -> nn.Module:
        """
        Apply hard pruning to the model.
        
        Args:
            model: Model with GatedConv2d/GatedLinear layers.
            remove_gates: If True, replace gated layers with standard layers
                         (producing a clean model without gate overhead).
            
        Returns:
            Pruned model (modified in-place).
        """
        pruned_model = copy.deepcopy(model)
        
        for name, module in pruned_model.named_modules():
            if isinstance(module, GatedConv2d):
                self._prune_gated_conv(module)
            elif isinstance(module, GatedLinear):
                self._prune_gated_linear(module)
        
        if remove_gates:
            pruned_model = self._remove_gates(pruned_model)
        
        return pruned_model
    
    def _prune_gated_conv(self, gated_conv: GatedConv2d):
        """Zero out channels with gate values below threshold."""
        gate_values = gated_conv.gate.get_gate_values()
        mask = (gate_values >= self.threshold).float()
        
        # Apply mask to conv weights: (out_channels, in_channels, H, W)
        with torch.no_grad():
            gated_conv.conv.weight.data *= mask.view(-1, 1, 1, 1)
            if gated_conv.conv.bias is not None:
                gated_conv.conv.bias.data *= mask
    
    def _prune_gated_linear(self, gated_linear: GatedLinear):
        """Zero out features with gate values below threshold."""
        gate_values = gated_linear.gate.get_gate_values()
        mask = (gate_values >= self.threshold).float()
        
        # Apply mask to linear weights: (out_features, in_features)
        with torch.no_grad():
            gated_linear.linear.weight.data *= mask.view(-1, 1)
            if gated_linear.linear.bias is not None:
                gated_linear.linear.bias.data *= mask
    
    def _remove_gates(self, model: nn.Module) -> nn.Module:
        """Replace GatedConv2d/GatedLinear with standard Conv2d/Linear."""
        for name, module in model.named_children():
            if isinstance(module, GatedConv2d):
                # Replace with standard Conv2d (weights already pruned)
                setattr(model, name, module.conv)
            elif isinstance(module, GatedLinear):
                setattr(model, name, module.linear)
            else:
                self._remove_gates(module)
        return model
    
    def get_pruning_report(self, model: nn.Module) -> Dict:
        """
        Generate a pruning report comparing before/after statistics.
        
        Args:
            model: Original model (before hard pruning).
            
        Returns:
            Report dict with parameter counts, sizes, and sparsity.
        """
        # Before pruning stats
        total_params_before = sum(p.numel() for p in model.parameters())
        size_before_mb = (total_params_before * 4) / (1024 * 1024)
        
        # Sparsity stats
        sparsity_stats = compute_model_sparsity(model, self.threshold)
        
        # After pruning (zero params don't count)
        pruned_model = self.prune(model, remove_gates=True)
        total_params_after = sum(p.numel() for p in pruned_model.parameters())
        
        # Count non-zero parameters
        nonzero_params = sum(
            (p != 0).sum().item() for p in pruned_model.parameters()
        )
        size_after_mb = (nonzero_params * 4) / (1024 * 1024)
        
        reduction_ratio = 1.0 - (nonzero_params / total_params_before)
        
        return {
            "total_params_before": total_params_before,
            "total_params_after": total_params_after,
            "nonzero_params_after": nonzero_params,
            "size_before_mb": size_before_mb,
            "size_after_mb": size_after_mb,
            "parameter_reduction": reduction_ratio,
            "sparsity": sparsity_stats,
            "threshold": self.threshold,
        }
