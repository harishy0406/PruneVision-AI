"""
PruneVision AI - CIFAR-10 Dataset
Dataset loading for CIFAR-10 with augmentation,
train/val/test splitting, and class-weighted sampling.
"""

import os
import random
from typing import Tuple, Dict, Optional, List
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms, datasets
from PIL import Image
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class CIFAR10Wrapper(Dataset):
    """
    Wrapper around CIFAR-10 dataset with optional transform.
    
    Args:
        cifar_dataset: Torchvision CIFAR10 dataset instance.
        transform (callable, optional): Transform to apply to images.
        class_names (list, optional): Ordered list of class names.
    """
    
    def __init__(
        self,
        cifar_dataset,
        transform: Optional[transforms.Compose] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.cifar_dataset = cifar_dataset
        self.transform = transform
        self.class_names = class_names or config.CLASS_NAMES
    
    def __len__(self) -> int:
        return len(self.cifar_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.cifar_dataset[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Return count of samples per class."""
        labels = [label for _, label in self.cifar_dataset]
        counter = Counter(labels)
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
        labels = [label for _, label in self.cifar_dataset]
        counter = Counter(labels)
        total = len(labels)
        num_classes = len(self.class_names)
        weights = torch.zeros(num_classes)
        for cls_idx in range(num_classes):
            count = counter.get(cls_idx, 1)
            weights[cls_idx] = total / (num_classes * count)
        return weights


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
    Create train/val/test DataLoaders for CIFAR-10.
    
    Args:
        data_dir: Root directory for CIFAR-10 data (defaults to ./data).
        batch_size: Batch size for DataLoader.
        num_workers: Number of worker processes.
        use_weighted_sampler: If True, use weighted random sampling for train.
        
    Returns:
        (train_loader, val_loader, test_loader, class_names, class_weights)
    """
    data_dir = data_dir or os.path.join(config.BASE_DIR, "data")
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers if num_workers is not None else config.NUM_WORKERS
    
    # Create data directory if needed
    os.makedirs(data_dir, exist_ok=True)
    
    # Define transforms
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")
    test_transform = get_transforms("test")
    
    # Load CIFAR-10
    print("[Dataset] Loading CIFAR-10...")
    cifar10_train = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=None  # We'll apply transforms per split
    )
    cifar10_test = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=None
    )
    
    # Split training set into train/val (80/20 of 50k = 40k/10k)
    train_size = int(0.8 * len(cifar10_train))
    val_size = len(cifar10_train) - train_size
    train_subset, val_subset = random_split(
        cifar10_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    print(f"[Dataset] Train: {train_size}, Val: {val_size}, Test: {len(cifar10_test)}")
    
    # Wrap datasets with transforms and class names
    train_dataset = CIFAR10Wrapper(
        train_subset,
        transform=train_transform,
        class_names=config.CLASS_NAMES,
    )
    val_dataset = CIFAR10Wrapper(
        val_subset,
        transform=val_transform,
        class_names=config.CLASS_NAMES,
    )
    test_dataset = CIFAR10Wrapper(
        cifar10_test,
        transform=test_transform,
        class_names=config.CLASS_NAMES,
    )
    
    # Get class distribution and weights
    print("[Dataset] Computing class weights...")
    all_labels = [label for _, label in cifar10_train]
    class_weights = torch.zeros(len(config.CLASS_NAMES))
    label_counts = Counter(all_labels)
    total = len(all_labels)
    num_classes = len(config.CLASS_NAMES)
    for cls_idx in range(num_classes):
        count = label_counts.get(cls_idx, 1)
        class_weights[cls_idx] = total / (num_classes * count)
    
    # Weighted sampler for training (handles class imbalance)
    train_sampler = None
    train_shuffle = True
    if use_weighted_sampler:
        train_labels = [label for _, label in train_subset.dataset]
        train_subset_indices = train_subset.indices
        train_subset_labels = [train_labels[idx] for idx in train_subset_indices]
        sample_weights = [class_weights[label].item() for label in train_subset_labels]
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
    
    return train_loader, val_loader, test_loader, config.CLASS_NAMES, class_weights
