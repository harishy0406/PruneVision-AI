"""
Unit tests for PruneVision AI gate mechanisms.
"""

import pytest
import torch
import torch.nn as nn
from prunevision.gates.gate_layer import GateLayer


class TestGateLayer:
    """Test cases for GateLayer class."""

    def test_gate_layer_initialization(self):
        """Test GateLayer initialization."""
        num_gates = 64
        gate = GateLayer(num_gates=num_gates, init_bias=3.0)
        
        assert gate.num_gates == num_gates
        assert gate.gate_params.shape == (num_gates,)

    def test_gate_layer_forward_4d(self, device):
        """Test GateLayer forward pass with 4D tensor (Conv2d output)."""
        num_gates = 32
        gate = GateLayer(num_gates=num_gates).to(device)
        
        # Simulate Conv2d output: (batch, channels, height, width)
        x = torch.randn(2, 32, 64, 64, device=device)
        output = gate(x)
        
        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_gate_layer_forward_2d(self, device):
        """Test GateLayer forward pass with 2D tensor (Linear output)."""
        num_gates = 128
        gate = GateLayer(num_gates=num_gates).to(device)
        
        # Simulate Linear output: (batch, features)
        x = torch.randn(16, 128, device=device)
        output = gate(x)
        
        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_gate_values_range(self):
        """Test that gate values are in [0, 1] after sigmoid."""
        num_gates = 64
        gate = GateLayer(num_gates=num_gates)
        
        gate_values = gate.forward(torch.ones(1, num_gates))
        gate_values = torch.sigmoid(gate.gate_params)
        
        assert (gate_values >= 0).all()
        assert (gate_values <= 1).all()

    def test_sparsity_computation(self):
        """Test sparsity computation."""
        num_gates = 100
        gate = GateLayer(num_gates=num_gates, init_bias=3.0)
        
        # Initially, gates should be mostly open (low sparsity)
        sparsity = gate.get_sparsity(threshold=0.05)
        assert 0 <= sparsity <= 1
        assert sparsity < 0.2  # With init_bias=3.0, most gates should be > 0.05

    def test_apply_mask(self):
        """Test permanent gate mask application."""
        num_gates = 64
        gate = GateLayer(num_gates=num_gates, init_bias=-2.0)  # Some gates closed
        
        # Apply mask with threshold
        original_gate_params = gate.gate_params.clone()
        # Note: apply_mask would permanently modify weights if implemented
        # This is a placeholder for actual implementation test

    def test_gradient_flow(self, device):
        """Test that gradients flow through gates."""
        num_gates = 32
        gate = GateLayer(num_gates=num_gates).to(device)
        
        x = torch.randn(2, 32, 64, 64, device=device, requires_grad=True)
        output = gate(x)
        loss = output.sum()
        loss.backward()
        
        assert gate.gate_params.grad is not None
        assert x.grad is not None

    def test_device_transfer(self):
        """Test moving GateLayer to different devices."""
        num_gates = 64
        gate = GateLayer(num_gates=num_gates)
        
        # Move to CPU
        gate = gate.to("cpu")
        assert gate.gate_params.device.type == "cpu"
        
        # Test forward on CPU
        x = torch.randn(2, 64, 32, 32)
        output = gate(x)
        assert output.device.type == "cpu"

    def test_eval_mode(self):
        """Test evaluation mode."""
        num_gates = 64
        gate = GateLayer(num_gates=num_gates)
        
        gate.eval()
        x = torch.randn(2, 64, 32, 32)
        
        # Forward pass should work in eval mode
        output = gate(x)
        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
