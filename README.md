# ✂️ PruneVision AI — Short Project Report

PruneVision AI is a PyTorch-based self-pruning vision framework built around learnable sigmoid gates. The project trains compact classifiers on CIFAR-10, applies L1 regularization to encourage sparse gates, and exports pruned models for efficient inference. The implementation is organized under [Full Implementation](Full%20Implementation), while the repository root stays focused on this report.

## Overview

The codebase is centered on three layers:

- **Modeling**: prunable MobileNetV3, ResNet-18, and EfficientNet-B0 backbones
- **Training**: a staged sparsity schedule that balances accuracy and compression
- **Deployment**: evaluation, pruning, ONNX export, and dashboard visualization

The best saved CIFAR-10 checkpoint in the repository is the MobileNetV3-Small run, which reports **92.37% test accuracy** and **45.3% sparsity**.

## Answers

### Why L1 on sigmoid gates encourages sparsity

Each gate is passed through a sigmoid, so its value stays in the range $[0, 1]$. Adding an L1 penalty on those gate values makes the optimizer pay a cost for keeping gates open. The easiest way to reduce that cost is to push unimportant gates toward 0, which shuts off channels and produces a sparse model.

### Result summary by $\lambda$

Only one completed run is stored in the current workspace, so the table below reflects the saved best MobileNetV3-Small checkpoint.

| Lambda | Test Accuracy | Sparsity Level (%) |
|-------:|--------------:|-------------------:|
| 0.0001 | 92.37 | 45.3 |

### Final gate distribution plot



## Snapshots and Demo

### Dashboard snapshots

![Overview snapshot](https://github.com/user-attachments/assets/81232db9-f76a-4e30-8c89-2215fd389ad2)

![Dataset explorer snapshot](https://github.com/user-attachments/assets/186c029e-676f-4d43-9a99-36e89d09b737)

![Training monitor snapshot](https://github.com/user-attachments/assets/fda92861-cf99-46bf-a9bf-722bf758aa63)

![Model analysis snapshot](https://github.com/user-attachments/assets/c444d36f-0ae1-4ef1-97bc-0fb7859cb1ee)

![Deployment snapshot](https://github.com/user-attachments/assets/c6b0e285-7961-4d1e-a303-a541a6b983f8)

### Project demo

[Watch the demo GIF](Full%20Implementation/Work%20Demo(Prunvison%20AI).gif)
