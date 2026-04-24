"""
PruneVision AI - Pruned Hybrid Model
Hybrid architecture combining MobileNetV3, ResNet-18, and EfficientNet-B0
with custom prunable layers for dynamic sparsity.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    EfficientNet_B0_Weights
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

from .base import PrunableModel
from ..gates.gate_wrapper import wrap_model_with_gates
from ..gates.gate_layer import GateLayer


class PrunedHybrid(PrunableModel):
    """
    Hybrid model combining features from MobileNetV3, ResNet-18, and EfficientNet-B0
    with custom prunable fusion layer.
    
    Architecture:
    - Three parallel backbones extracting features
    - Feature concatenation with adaptive pooling for compatibility
    - Custom prunable fusion layer with learnable gating
    - Final classifier head
    
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
        
        # Load pretrained backbones
        if pretrained:
            mob_weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            res_weights = ResNet18_Weights.IMAGENET1K_V1
            eff_weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            mob_weights = res_weights = eff_weights = None
        
        # MobileNetV3-Small backbone (remove classifier)
        self.mobile_net = models.mobilenet_v3_small(weights=mob_weights)
        self.mobile_net.classifier = nn.Identity()  # Remove classifier to get features
        
        # ResNet-18 backbone (remove avgpool and fc)
        self.resnet = models.resnet18(weights=res_weights)
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()
        
        # EfficientNet-B0 backbone (remove classifier)
        self.efficientnet = models.efficientnet_b0(weights=eff_weights)
        self.efficientnet.classifier = nn.Identity()
        
        # Adaptive pooling to standardize feature dimensions (target 1x1x512 for each)
        self.mobile_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.eff_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Wrap backbones with gates
        wrap_model_with_gates(self.mobile_net, gate_init_bias=gate_init_bias, skip_patterns=["classifier"])
        wrap_model_with_gates(self.resnet, gate_init_bias=gate_init_bias, skip_patterns=["avgpool", "fc"])
        wrap_model_with_gates(self.efficientnet, gate_init_bias=gate_init_bias, skip_patterns=["classifier"])
        
        # Custom prunable fusion layer
        # Total features: 1024 (mob) + 512 (res) + 1280 (eff) = 2816
        fusion_in_features = 1024 + 512 + 1280  # Adjust based on actual outputs
        self.fusion_gate = GateLayer(num_gates=fusion_in_features, init_bias=gate_init_bias)
        self.fusion_linear = nn.Linear(fusion_in_features, 512)  # Prunable fusion
        
        # Final classifier
        self.classifier = nn.Linear(512, num_classes)
        
        # Optionally freeze backbones
        if freeze_backbone:
            self.freeze_backbones(True)
    
    def freeze_backbones(self, freeze: bool = True):
        """Freeze/unfreeze backbone parameters."""
        for param in self.mobile_net.parameters():
            param.requires_grad = not freeze
        for param in self.resnet.parameters():
            param.requires_grad = not freeze
        for param in self.efficientnet.parameters():
            param.requires_grad = not freeze
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features from each backbone
        mob_feat = self.mobile_net(x)  # [B, 1024, H, W]
        mob_feat = self.mobile_pool(mob_feat).flatten(1)  # [B, 1024]
        
        res_feat = self.resnet(x)  # [B, 512, H, W]
        res_feat = self.resnet_pool(res_feat).flatten(1)  # [B, 512]
        
        eff_feat = self.efficientnet(x)  # [B, 1280, H, W]
        eff_feat = self.eff_pool(eff_feat).flatten(1)  # [B, 1280]
        
        # Concatenate features
        combined = torch.cat([mob_feat, res_feat, eff_feat], dim=1)  # [B, 2816]
        
        # Apply custom prunable fusion
        gated_combined = self.fusion_gate(combined.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)  # Gate the features
        fusion_out = self.fusion_linear(gated_combined)  # [B, 512]
        
        # Final classification
        return self.classifier(fusion_out)
    
    def get_model_name(self) -> str:
        return "PrunedHybrid-MobileNet-ResNet-EfficientNet"