"""
PruneVision AI - Retail Product Dataset
Custom dataset for loading retail product images with augmentation,
stratified train/val/test splitting, and class-weighted sampling.
"""

import os
import random
from typing import Tuple, Dict, Optional, List
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class RetailDataset(Dataset):
    """
    Retail product image dataset.
    
    Loads images from a directory structure where each subdirectory is a class:
        data/images/BEANS/001.png
        data/images/CAKE/001.png
        ...
    
    Args:
        image_paths (list): List of image file paths.
        labels (list): Corresponding integer labels.
        transform (callable, optional): Transform to apply to images.
        class_names (list, optional): Ordered list of class names.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or config.CLASS_NAMES
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Return count of samples per class."""
        counter = Counter(self.labels)
        return {
            self.class_names[k]: v 
            for k, v in sorted(counter.items())
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for handling imbalanced data.
        
        Returns:
            Tensor of shape (num_classes,) with weight per class.
        """
        counter = Counter(self.labels)
        total = len(self.labels)
        num_classes = len(self.class_names)
        weights = torch.zeros(num_classes)
        for cls_idx in range(num_classes):
            count = counter.get(cls_idx, 1)
            weights[cls_idx] = total / (num_classes * count)
        return weights


def _load_image_paths_and_labels(
    data_dir: str,
    class_names: Optional[List[str]] = None,
) -> Tuple[List[str], List[int], List[str]]:
    """
    Scan data directory and collect image paths with labels.
    
    Args:
        data_dir: Root directory containing class subdirectories.
        class_names: Optional predetermined class ordering.
        
    Returns:
        (image_paths, labels, class_names)
    """
    if class_names is None:
        # Sort for deterministic ordering
        class_names = sorted([
            d for d in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, d))
        ])
    
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    image_paths = []
    labels = []
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_to_idx[class_name])
    
    return image_paths, labels, class_names


def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Get image transforms for a given split.
    
    Args:
        split: One of "train", "val", "test".
        
    Returns:
        Composed transform pipeline.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(
                config.IMAGE_SIZE,
                scale=config.AUGMENTATION["random_resized_crop"]["scale"],
            ),
            transforms.RandomHorizontalFlip(
                p=config.AUGMENTATION["horizontal_flip_prob"]
            ),
            transforms.ColorJitter(
                brightness=config.AUGMENTATION["color_jitter"]["brightness"],
                contrast=config.AUGMENTATION["color_jitter"]["contrast"],
                saturation=config.AUGMENTATION["color_jitter"]["saturation"],
                hue=config.AUGMENTATION["color_jitter"]["hue"],
            ),
            transforms.RandomRotation(config.AUGMENTATION["rotation_degrees"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.IMAGENET_MEAN,
                std=config.IMAGENET_STD,
            ),
        ])
    else:
        # Validation / Test: deterministic resize + center crop
        return transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE + 32),  # Resize larger
            transforms.CenterCrop(config.IMAGE_SIZE),    # Center crop to exact size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.IMAGENET_MEAN,
                std=config.IMAGENET_STD,
            ),
        ])


def get_dataloaders(
    data_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], torch.Tensor]:
    """
    Create train/val/test DataLoaders with stratified splitting.
    
    Args:
        data_dir: Root directory with class subdirectories.
        batch_size: Batch size for DataLoader.
        num_workers: Number of worker processes.
        use_weighted_sampler: If True, use weighted random sampling for train.
        
    Returns:
        (train_loader, val_loader, test_loader, class_names, class_weights)
    """
    data_dir = data_dir or config.DATA_DIR
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers if num_workers is not None else config.NUM_WORKERS
    
    # Load all paths and labels
    image_paths, labels, class_names = _load_image_paths_and_labels(
        data_dir, config.CLASS_NAMES
    )
    
    print(f"[Dataset] Found {len(image_paths)} images across {len(class_names)} classes")
    
    # Stratified train/val/test split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
        stratify=labels,
        random_state=config.RANDOM_SEED,
    )
    
    # Split remaining into val and test
    relative_test_size = config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=relative_test_size,
        stratify=temp_labels,
        random_state=config.RANDOM_SEED,
    )
    
    print(f"[Dataset] Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Create datasets
    train_dataset = RetailDataset(
        train_paths, train_labels,
        transform=get_transforms("train"),
        class_names=class_names,
    )
    val_dataset = RetailDataset(
        val_paths, val_labels,
        transform=get_transforms("val"),
        class_names=class_names,
    )
    test_dataset = RetailDataset(
        test_paths, test_labels,
        transform=get_transforms("test"),
        class_names=class_names,
    )
    
    # Class weights for loss function
    class_weights = train_dataset.get_class_weights()
    
    # Weighted sampler for training (handles class imbalance)
    train_sampler = None
    train_shuffle = True
    if use_weighted_sampler:
        sample_weights = [class_weights[label].item() for label in train_labels]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False  # Sampler handles shuffling
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    return train_loader, val_loader, test_loader, class_names, class_weights
