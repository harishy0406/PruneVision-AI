"""
PruneVision AI - Gate Wrapper
Wraps standard PyTorch layers (Conv2d, Linear) with learnable gates.
Provides utilities to automatically instrument an existing model with gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from .gate_layer import GateLayer


class GatedConv2d(nn.Module):
    """
    Conv2d layer with per-output-channel gating.
    
    Wraps an existing nn.Conv2d and adds a GateLayer that controls
    which output channels are active. During training, unimportant
    channels are driven toward zero via L1 regularization on gates.
    
    Args:
        conv_layer (nn.Conv2d): Existing convolutional layer to wrap.
        gate_init_bias (float): Initial gate bias (positive = gates start open).
    """
    
    def __init__(self, conv_layer: nn.Conv2d, gate_init_bias: float = 3.0):
        super().__init__()
        self.conv = conv_layer
        self.gate = GateLayer(
            num_gates=conv_layer.out_channels,
            init_bias=gate_init_bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: conv → gate masking."""
        out = self.conv(x)
        return self.gate(out)
    
    @property
    def weight(self):
        return self.conv.weight
    
    @property
    def bias(self):
        return self.conv.bias


class GatedLinear(nn.Module):
    """
    Linear layer with per-output-feature gating.
    
    Wraps an existing nn.Linear and adds a GateLayer that controls
    which output features are active.
    
    Args:
        linear_layer (nn.Linear): Existing linear layer to wrap.
        gate_init_bias (float): Initial gate bias.
    """
    
    def __init__(self, linear_layer: nn.Linear, gate_init_bias: float = 3.0):
        super().__init__()
        self.linear = linear_layer
        self.gate = GateLayer(
            num_gates=linear_layer.out_features,
            init_bias=gate_init_bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: linear → gate masking."""
        out = self.linear(x)
        return self.gate(out)
    
    @property
    def weight(self):
        return self.linear.weight
    
    @property
    def bias(self):
        return self.linear.bias


def wrap_model_with_gates(
    model: nn.Module,
    gate_init_bias: float = 3.0,
    skip_last_linear: bool = True,
    skip_patterns: Optional[List[str]] = None,
) -> nn.Module:
    """
    Recursively wrap Conv2d and Linear layers in a model with gated variants.
    
    This function traverses the model tree and replaces:
    - nn.Conv2d → GatedConv2d
    - nn.Linear → GatedLinear
    
    The final classifier layer is optionally skipped (we don't want to prune
    the output layer since it directly maps to classes).
    
    Args:
        model: PyTorch model to instrument with gates.
        gate_init_bias: Initial bias for gate parameters.
        skip_last_linear: If True, don't gate the last Linear layer (classifier head).
        skip_patterns: List of layer name patterns to skip (e.g., ["classifier", "fc"]).
        
    Returns:
        Model with gated layers (modified in-place and returned).
    """
    if skip_patterns is None:
        skip_patterns = []
    
    # Find all linear layers to identify the last one
    all_linears = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            all_linears.append(name)
    
    last_linear_name = all_linears[-1] if all_linears else None
    
    def _should_skip(name: str) -> bool:
        """Check if a layer should be skipped based on patterns."""
        for pattern in skip_patterns:
            if pattern in name:
                return True
        if skip_last_linear and name == last_linear_name:
            return True
        return False
    
    def _wrap_recursive(module: nn.Module, prefix: str = ""):
        """Recursively replace layers with gated variants."""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Conv2d) and not _should_skip(full_name):
                setattr(module, name, GatedConv2d(child, gate_init_bias))
            elif isinstance(child, nn.Linear) and not _should_skip(full_name):
                setattr(module, name, GatedLinear(child, gate_init_bias))
            else:
                # Recurse into child modules
                _wrap_recursive(child, full_name)
    
    _wrap_recursive(model)
    return model


def collect_gate_layers(model: nn.Module) -> List[Tuple[str, GateLayer]]:
    """
    Collect all GateLayer instances from a model.
    
    Returns:
        List of (name, GateLayer) tuples.
    """
    gates = []
    for name, module in model.named_modules():
        if isinstance(module, GateLayer):
            gates.append((name, module))
    return gates


def compute_total_gate_l1(model: nn.Module) -> torch.Tensor:
    """
    Compute total L1 regularization loss across all gates in the model.
    
    Returns:
        Scalar tensor: sum of L1 losses from all GateLayer instances.
    """
    total_l1 = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, module in model.named_modules():
        if isinstance(module, GateLayer):
            total_l1 = total_l1 + module.get_l1_loss()
    return total_l1


def compute_model_sparsity(model: nn.Module, threshold: float = 0.05) -> dict:
    """
    Compute sparsity statistics across all gated layers.
    
    Returns:
        Dictionary with per-layer and global sparsity info.
    """
    layer_stats = {}
    total_gates = 0
    total_pruned = 0
    
    for name, module in model.named_modules():
        if isinstance(module, GateLayer):
            sparsity = module.get_sparsity(threshold)
            active = module.get_active_count(threshold)
            layer_stats[name] = {
                "num_gates": module.num_gates,
                "active": active,
                "pruned": module.num_gates - active,
                "sparsity": sparsity,
                "mean_gate": module.get_gate_values().mean().item(),
            }
            total_gates += module.num_gates
            total_pruned += (module.num_gates - active)
    
    global_sparsity = total_pruned / total_gates if total_gates > 0 else 0.0
    
    return {
        "layers": layer_stats,
        "total_gates": total_gates,
        "total_pruned": total_pruned,
        "total_active": total_gates - total_pruned,
        "global_sparsity": global_sparsity,
    }
