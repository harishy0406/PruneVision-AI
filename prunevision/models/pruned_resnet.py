"""
PruneVision AI - Pruned ResNet-18
ResNet-18 with learnable channel-wise gates for self-pruning.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

from .base import PrunableModel
from ..gates.gate_wrapper import wrap_model_with_gates


class PrunedResNet18(PrunableModel):
    """
    ResNet-18 with self-pruning gates.
    
    Architecture: ResNet-18 backbone + gated convolutions.
    - Pretrained on ImageNet for transfer learning
    - Channel-wise gates on all Conv2d layers
    - Custom FC head for retail products
    
    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Use ImageNet pretrained weights.
        gate_init_bias (float): Initial gate bias.
        freeze_backbone (bool): Freeze early layers.
    """
    
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = config.PRETRAINED,
        gate_init_bias: float = config.GATE_INIT_BIAS,
        freeze_backbone: bool = config.FREEZE_EARLY_LAYERS,
    ):
        super().__init__(num_classes, pretrained)
        
        # Load pretrained ResNet-18
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Replace classifier for our number of classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        # Wrap Conv2d layers with gates (skip classifier FC)
        wrap_model_with_gates(
            self.backbone,
            gate_init_bias=gate_init_bias,
            skip_last_linear=True,
        )
        
        if freeze_backbone:
            self.freeze_backbone(True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_model_name(self) -> str:
        return "PrunedResNet-18"
