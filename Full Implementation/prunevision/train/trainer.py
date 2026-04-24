"""
PruneVision AI - Training Engine
Main training loop with integrated self-pruning, sparsity scheduling,
and comprehensive logging.
"""

import os
import json
import time
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

from .scheduler import SparsityScheduler
from .metrics import compute_metrics, compute_topk_accuracy, measure_inference_latency
from ..gates.gate_wrapper import compute_model_sparsity


class PruneVisionTrainer:
    """
    Training engine with integrated self-pruning.
    
    Handles the complete training lifecycle:
    - Joint optimization of weights and gate parameters
    - 3-stage sparsity scheduling
    - Class-weighted cross-entropy loss
    - Validation with sparsity monitoring
    - Checkpoint saving and training history
    
    Args:
        model: PrunableModel instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        class_weights: Tensor of per-class weights for loss.
        class_names: List of class name strings.
        device: Device to train on.
        config_override: Optional dict to override config values.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        class_names: Optional[List[str]] = None,
        device: str = "cpu",
        config_override: Optional[Dict] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names or config.CLASS_NAMES
        self.device = device
        
        # Config
        cfg = config_override or {}
        self.epochs = cfg.get("epochs", config.EPOCHS)
        self.lr = cfg.get("lr", config.LEARNING_RATE)
        self.weight_decay = cfg.get("weight_decay", config.WEIGHT_DECAY)
        
        # Loss function with class weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(device)
            )
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Learning rate scheduler
        if config.LR_SCHEDULER == "cosine":
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=1e-6
            )
        elif config.LR_SCHEDULER == "step":
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.LR_STEP_SIZE,
                gamma=config.LR_GAMMA,
            )
        else:
            self.lr_scheduler = None
        
        # Sparsity scheduler
        self.sparsity_scheduler = SparsityScheduler()
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "sparsity": [],
            "lambda": [],
            "lr": [],
            "epoch_time": [],
            "stage": [],
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train(self, save_dir: Optional[str] = None) -> Dict:
        """
        Run full training loop.
        
        Args:
            save_dir: Directory to save checkpoints and history.
            
        Returns:
            Training history dictionary.
        """
        save_dir = save_dir or config.CHECKPOINT_DIR
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"PruneVision AI Training")
        print(f"{'='*60}")
        print(f"Model: {self.model.get_model_name()}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"Learning Rate: {self.lr}")
        print(f"Batch Size: {self.train_loader.batch_size}")
        print(f"Train Samples: {len(self.train_loader.dataset)}")
        print(f"Val Samples: {len(self.val_loader.dataset)}")
        print(f"\n{self.sparsity_scheduler.get_schedule_summary()}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Get current sparsity lambda
            current_lambda = self.sparsity_scheduler.get_lambda(epoch)
            current_stage = self.sparsity_scheduler.get_current_stage(epoch)
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Training epoch
            train_loss, train_acc = self._train_epoch(epoch, current_lambda)
            
            # Validation epoch
            val_loss, val_acc = self._validate_epoch(epoch)
            
            # Get sparsity
            sparsity_stats = compute_model_sparsity(self.model)
            global_sparsity = sparsity_stats["global_sparsity"]
            
            # Update LR scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Log history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["sparsity"].append(global_sparsity)
            self.history["lambda"].append(current_lambda)
            self.history["lr"].append(current_lr)
            self.history["epoch_time"].append(epoch_time)
            self.history["stage"].append(current_stage)
            
            # Print epoch summary
            print(
                f"Epoch {epoch+1:3d}/{self.epochs} | "
                f"Stage: {current_stage:12s} | "
                f"L: {current_lambda:.4f} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2%} | "
                f"Val Acc: {val_acc:.2%} | "
                f"Sparsity: {global_sparsity:.1%} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self._save_checkpoint(
                    os.path.join(save_dir, "best_model.pth"),
                    epoch, val_acc, global_sparsity,
                )
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(
                    os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"),
                    epoch, val_acc, global_sparsity,
                )
        
        # Save final model and history
        self._save_checkpoint(
            os.path.join(save_dir, "final_model.pth"),
            self.epochs - 1, val_acc, global_sparsity,
        )
        self._save_history(os.path.join(save_dir, "training_history.json"))
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Val Accuracy: {self.best_val_acc:.2%} (Epoch {self.best_epoch+1})")
        print(f"Final Sparsity: {global_sparsity:.1%}")
        print(f"{'='*60}\n")
        
        return self.history
    
    def _train_epoch(self, epoch: int, sparsity_lambda: float) -> Tuple[float, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"  Train Epoch {epoch+1}",
            leave=False,
            ncols=100,
        )
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Classification loss
            ce_loss = self.criterion(outputs, labels)
            
            # Gate L1 regularization loss
            gate_l1 = self.model.get_gate_l1_loss()
            
            # Total loss = CE + lambda * L1(gates)
            total_batch_loss = ce_loss + sparsity_lambda * gate_l1
            
            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += ce_loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                "loss": f"{ce_loss.item():.3f}",
                "acc": f"{correct/total:.2%}",
            })
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Tuple[float, float]:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Full evaluation on test set with detailed metrics.
        
        Args:
            test_loader: Test DataLoader.
            
        Returns:
            Dictionary with all evaluation metrics.
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_outputs = []
        
        for images, labels in tqdm(test_loader, desc="  Evaluating", leave=False):
            images = images.to(self.device)
            outputs = self.model(images)
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_outputs.append(outputs.cpu())
        
        # Compute metrics
        metrics = compute_metrics(all_preds, all_labels, self.class_names)
        
        # Top-k accuracy
        all_outputs = torch.cat(all_outputs)
        all_labels_tensor = torch.tensor(all_labels)
        topk = compute_topk_accuracy(all_outputs, all_labels_tensor, topk=(1, 5))
        metrics.update(topk)
        
        # Sparsity
        sparsity_stats = compute_model_sparsity(self.model)
        metrics["sparsity"] = sparsity_stats
        
        # Model stats
        metrics["total_params"] = self.model.get_total_params()
        metrics["model_size_mb"] = self.model.get_model_size_mb()
        
        # Inference latency
        latency = measure_inference_latency(self.model, device=self.device)
        metrics["latency"] = latency
        
        return metrics
    
    def _save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_acc: float,
        sparsity: float,
    ):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "sparsity": sparsity,
            "model_name": self.model.get_model_name(),
            "num_classes": self.model.num_classes,
            "config": {
                "lr": self.lr,
                "epochs": self.epochs,
                "batch_size": self.train_loader.batch_size,
                "image_size": config.IMAGE_SIZE,
            },
        }, path)
    
    def _save_history(self, path: str):
        """Save training history to JSON."""
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {path}")
