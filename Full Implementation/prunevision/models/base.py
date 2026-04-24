"""
PruneVision AI - Base Prunable Model
Abstract base class providing common interface for all prunable models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional

from ..gates.gate_wrapper import (
    collect_gate_layers,
    compute_total_gate_l1,
    compute_model_sparsity,
)


class PrunableModel(nn.Module, ABC):
    """
    Abstract base class for models with self-pruning capability.
    
    Provides common interface for:
    - Gate L1 regularization loss
    - Sparsity computation
    - Parameter counting (total vs active)
    - Hard pruning (permanent removal of gated-out weights)
    """
    
    def __init__(self, num_classes: int = 25, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self._pretrained = pretrained
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return human-readable model name."""
        pass
    
    def get_gate_l1_loss(self) -> torch.Tensor:
        """Compute total L1 regularization loss across all gates."""
        return compute_total_gate_l1(self)
    
    def get_sparsity_stats(self, threshold: float = 0.05) -> Dict:
        """Get detailed sparsity statistics per layer and globally."""
        return compute_model_sparsity(self, threshold)
    
    def get_global_sparsity(self, threshold: float = 0.05) -> float:
        """Get overall model sparsity ratio."""
        stats = self.get_sparsity_stats(threshold)
        return stats["global_sparsity"]
    
    def get_total_params(self) -> int:
        """Count total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_gate_layers(self):
        """Collect all GateLayer instances."""
        return collect_gate_layers(self)
    
    def get_model_size_mb(self) -> float:
        """Estimate model size in MB (float32 parameters)."""
        total_params = self.get_total_params()
        return (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    
    def freeze_backbone(self, freeze: bool = True):
        """
        Freeze/unfreeze backbone (all layers except classifier and gates).
        Gates are always trainable.
        """
        from ..gates.gate_layer import GateLayer
        
        for name, param in self.named_parameters():
            # Always keep gates trainable
            is_gate = False
            for module_name, module in self.named_modules():
                if isinstance(module, GateLayer) and name.startswith(module_name):
                    is_gate = True
                    break
            
            # Keep classifier trainable
            is_classifier = "classifier" in name or "fc" in name or "head" in name
            
            if not is_gate and not is_classifier:
                param.requires_grad = not freeze
    
    def summary(self) -> str:
        """Return a summary string of the model."""
        total = self.get_total_params()
        trainable = self.get_trainable_params()
        sparsity = self.get_global_sparsity()
        size_mb = self.get_model_size_mb()
        
        return (
            f"{'='*50}\n"
            f"Model: {self.get_model_name()}\n"
            f"Classes: {self.num_classes}\n"
            f"Total Parameters: {total:,}\n"
            f"Trainable Parameters: {trainable:,}\n"
            f"Model Size: {size_mb:.2f} MB\n"
            f"Global Sparsity: {sparsity:.1%}\n"
            f"{'='*50}"
        )
