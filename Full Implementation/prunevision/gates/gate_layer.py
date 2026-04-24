"""
PruneVision AI - Gate Layer
Core learnable gating mechanism for self-pruning neural networks.

Each gate parameter controls the importance of a corresponding weight/channel.
During training, gates are optimized jointly with weights via L1 regularization,
driving unimportant connections toward zero.
"""

import torch
import torch.nn as nn


class GateLayer(nn.Module):
    """
    Learnable gating mechanism for neural network pruning.
    
    Maintains gate parameters that are multiplied with weights during forward pass.
    Gates use sigmoid activation to produce values in [0, 1].
    L1 regularization on gate values encourages sparsity.
    
    Args:
        num_gates (int): Number of gate parameters (typically = number of output channels)
        init_bias (float): Initial bias for gate parameters. Positive values mean
                          gates start "open" (sigmoid(3.0) ≈ 0.95).
    """
    
    def __init__(self, num_gates: int, init_bias: float = 3.0):
        super().__init__()
        self.num_gates = num_gates
        # Initialize gates with positive bias so they start "open"
        self.gate_params = nn.Parameter(torch.full((num_gates,), init_bias))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply gating mask to input tensor.
        
        For convolutional layers: x shape is (batch, channels, H, W)
        Gate is applied per-channel: output = x * sigmoid(gate).view(1, -1, 1, 1)
        
        For linear layers: x shape is (batch, features) or (features,)
        Gate is applied per-feature: output = x * sigmoid(gate)
        """
        gate_values = torch.sigmoid(self.gate_params)
        
        if x.dim() == 4:
            # Conv2d output: (B, C, H, W) — gate per channel
            return x * gate_values.view(1, -1, 1, 1)
        elif x.dim() == 2:
            # Linear output: (B, F) — gate per feature
            return x * gate_values.view(1, -1)
        elif x.dim() == 1:
            # Single vector
            return x * gate_values
        else:
            # Fallback: broadcast
            return x * gate_values
    
    def get_gate_values(self) -> torch.Tensor:
        """Return sigmoid-activated gate values (importance scores in [0, 1])."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_params)
    
    def get_sparsity(self, threshold: float = 0.05) -> float:
        """
        Compute fraction of gates below threshold (fraction of 'pruned' channels).
        
        Args:
            threshold: Gates with sigmoid(g) < threshold are considered pruned.
            
        Returns:
            Sparsity ratio in [0, 1]. Higher = more sparse.
        """
        gate_values = self.get_gate_values()
        num_pruned = (gate_values < threshold).sum().item()
        return num_pruned / self.num_gates
    
    def get_l1_loss(self) -> torch.Tensor:
        """Compute L1 regularization loss on gate values to encourage sparsity."""
        return torch.sigmoid(self.gate_params).sum()
    
    def get_active_count(self, threshold: float = 0.05) -> int:
        """Return number of gates above threshold (active channels)."""
        gate_values = self.get_gate_values()
        return (gate_values >= threshold).sum().item()
    
    def apply_mask(self, threshold: float = 0.05) -> torch.Tensor:
        """
        Create a binary mask based on gate threshold.
        
        Returns:
            Binary tensor: 1 for active gates, 0 for pruned gates.
        """
        gate_values = self.get_gate_values()
        return (gate_values >= threshold).float()
    
    def extra_repr(self) -> str:
        gate_values = self.get_gate_values()
        mean_val = gate_values.mean().item()
        sparsity = self.get_sparsity()
        return (
            f"num_gates={self.num_gates}, "
            f"mean_gate={mean_val:.3f}, "
            f"sparsity={sparsity:.1%}"
        )
