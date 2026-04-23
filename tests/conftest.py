"""
Test configuration and fixtures for PruneVision AI test suite.
"""

import os
import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test artifacts."""
    return tmp_path


@pytest.fixture
def device():
    """Provide device for tests."""
    return "cpu"  # Always use CPU for tests


@pytest.fixture
def sample_image_batch():
    """Create a sample batch of images for testing."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def sample_labels():
    """Create sample labels for testing."""
    return torch.tensor([0, 1])


@pytest.fixture
def num_classes():
    """Number of classes for testing."""
    return 25


@pytest.fixture
def model_config():
    """Configuration for model testing."""
    return {
        "num_classes": 25,
        "pretrained": False,
        "gate_init_bias": 3.0,
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
