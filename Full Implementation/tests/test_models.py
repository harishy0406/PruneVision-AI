"""
Unit tests for PruneVision AI models.
"""

import pytest
import torch
from prunevision.models import PrunedMobileNetV3, PrunedResNet18, PrunedEfficientNetB0


class TestPrunedMobileNetV3:
    """Test cases for PrunedMobileNetV3 model."""

    def test_model_initialization(self):
        """Test model can be initialized."""
        model = PrunedMobileNetV3(num_classes=25, pretrained=False)
        assert model is not None
        assert model.num_classes == 25

    def test_forward_pass(self, sample_image_batch, device):
        """Test forward pass through model."""
        model = PrunedMobileNetV3(num_classes=25, pretrained=False).to(device)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_image_batch.to(device))
        
        assert output.shape == (2, 25)  # batch_size=2, num_classes=25

    def test_parameter_count(self):
        """Test parameter counting."""
        model = PrunedMobileNetV3(num_classes=25, pretrained=False)
        total_params = model.get_total_params()
        
        assert total_params > 0
        assert isinstance(total_params, int)

    def test_sparsity_computation(self):
        """Test sparsity statistics."""
        model = PrunedMobileNetV3(num_classes=25, pretrained=False)
        sparsity_stats = model.get_sparsity_stats()
        
        assert "global_sparsity" in sparsity_stats
        assert 0 <= sparsity_stats["global_sparsity"] <= 1


class TestPrunedResNet18:
    """Test cases for PrunedResNet18 model."""

    def test_model_initialization(self):
        """Test model can be initialized."""
        model = PrunedResNet18(num_classes=25, pretrained=False)
        assert model is not None
        assert model.num_classes == 25

    def test_forward_pass(self, sample_image_batch, device):
        """Test forward pass through model."""
        model = PrunedResNet18(num_classes=25, pretrained=False).to(device)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_image_batch.to(device))
        
        assert output.shape == (2, 25)


class TestPrunedEfficientNetB0:
    """Test cases for PrunedEfficientNetB0 model."""

    def test_model_initialization(self):
        """Test model can be initialized."""
        model = PrunedEfficientNetB0(num_classes=25, pretrained=False)
        assert model is not None
        assert model.num_classes == 25

    def test_forward_pass(self, sample_image_batch, device):
        """Test forward pass through model."""
        model = PrunedEfficientNetB0(num_classes=25, pretrained=False).to(device)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_image_batch.to(device))
        
        assert output.shape == (2, 25)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
