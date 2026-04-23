"""
Unit tests for PruneVision AI data pipeline.
"""

import pytest
import torch
import os
from prunevision.data.dataset import RetailDataset
import config


class TestRetailDataset:
    """Test cases for RetailDataset."""

    def test_dataset_initialization(self, temp_dir):
        """Test dataset can be initialized."""
        # Create dummy data
        image_paths = []
        labels = []
        
        dataset = RetailDataset(
            image_paths=image_paths,
            labels=labels,
            class_names=config.CLASS_NAMES
        )
        
        assert dataset is not None
        assert len(dataset) == 0

    def test_class_names(self):
        """Test class names are correct."""
        assert len(config.CLASS_NAMES) == 25
        assert "BEANS" in config.CLASS_NAMES
        assert "WATER" in config.CLASS_NAMES

    def test_get_class_distribution(self):
        """Test class distribution computation."""
        image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
        labels = [0, 0, 1]
        
        dataset = RetailDataset(
            image_paths=image_paths,
            labels=labels,
            class_names=config.CLASS_NAMES
        )
        
        dist = dataset.get_class_distribution()
        assert isinstance(dist, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
