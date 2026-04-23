"""
PruneVision AI - Pruned MobileNetV3-Small
MobileNetV3-Small with learnable channel-wise gates for self-pruning.
Primary model for CPU training due to small parameter count (~2.5M).
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

from .base import PrunableModel
from ..gates.gate_wrapper import wrap_model_with_gates


class PrunedMobileNetV3(PrunableModel):
    """
    MobileNetV3-Small with self-pruning gates.
    
    Architecture: MobileNetV3-Small backbone + gated convolutions.
    - Pretrained on ImageNet for transfer learning
    - Channel-wise gates on all Conv2d layers
    - Custom classifier head for retail products
    
    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Use ImageNet pretrained weights.
        gate_init_bias (float): Initial gate bias.
        freeze_backbone (bool): Freeze early layers for transfer learning.
    """
    
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = config.PRETRAINED,
        gate_init_bias: float = config.GATE_INIT_BIAS,
        freeze_backbone: bool = config.FREEZE_EARLY_LAYERS,
    ):
        super().__init__(num_classes, pretrained)
        
        # Load pretrained MobileNetV3-Small
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.backbone = models.mobilenet_v3_small(weights=weights)
        else:
            self.backbone = models.mobilenet_v3_small(weights=None)
        
        # Replace classifier for our number of classes
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)
        
        # Wrap all Conv2d layers with gates (skip final classifier Linear)
        wrap_model_with_gates(
            self.backbone,
            gate_init_bias=gate_init_bias,
            skip_last_linear=True,
            skip_patterns=["classifier"],
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            self.freeze_backbone(True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_model_name(self) -> str:
        return "PrunedMobileNetV3-Small"
