import numpy as np
import torch
import matplotlib.pyplot as plt

from prunevision.models.pruned_mobilenet import PrunedMobileNetV3
from prunevision.gates.gate_wrapper import collect_gate_layers

ckpt = torch.load(
    "outputs/checkpoints/mobilenetv3_small/best_model.pth",
    map_location="cpu",
)

model = PrunedMobileNetV3(num_classes=10, pretrained=False)
model.load_state_dict(ckpt["model_state_dict"], strict=False)

gate_values = []
for _, gate_layer in collect_gate_layers(model):
    gate_values.extend(gate_layer.get_gate_values().cpu().numpy().ravel())

gate_values = np.array(gate_values)

plt.figure(figsize=(8, 5))
plt.hist(gate_values, bins=50, color="#667eea", edgecolor="black", alpha=0.85)
plt.axvline(0.05, color="red", linestyle="--", linewidth=2, label="Pruning threshold")
plt.title("Final Gate Value Distribution")
plt.xlabel("Gate value")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()