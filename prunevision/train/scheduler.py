"""
PruneVision AI - Sparsity Scheduler
3-stage sparsity lambda scheduling as defined in the PRD:
  Stage 1 (Warm-up): Low penalty, network learns representations
  Stage 2 (Progressive): Increasing penalty, drives redundant connections to zero
  Stage 3 (Fine-tuning): No penalty, polishes accuracy with fixed sparse structure
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class SparsityScheduler:
    """
    Manages the sparsity regularization coefficient (lambda) across training epochs.
    
    The scheduler implements a 3-stage approach:
    - Stage 1: Low lambda for warm-up (network learns features first)
    - Stage 2: Progressive lambda increase (drives pruning)
    - Stage 3: Zero lambda for fine-tuning (accuracy recovery)
    
    Args:
        schedule (dict): Stage definitions from config.SPARSITY_SCHEDULE.
        total_epochs (int): Total number of training epochs.
    """
    
    def __init__(
        self,
        schedule: dict = None,
        total_epochs: int = None,
    ):
        self.schedule = schedule or config.SPARSITY_SCHEDULE
        self.total_epochs = total_epochs or config.EPOCHS
        self.current_epoch = 0
        self.current_lambda = 0.0
        self.history = []
    
    def get_lambda(self, epoch: int) -> float:
        """
        Compute the sparsity penalty coefficient for a given epoch.
        
        Args:
            epoch: Current epoch number (0-indexed).
            
        Returns:
            Lambda value (sparsity penalty coefficient).
        """
        self.current_epoch = epoch
        
        for stage_key, stage in self.schedule.items():
            start = stage["start_epoch"]
            end = stage["end_epoch"]
            
            if start <= epoch < end:
                # Linear interpolation within stage
                if end > start:
                    progress = (epoch - start) / (end - start)
                else:
                    progress = 0.0
                
                lambda_val = (
                    stage["lambda_start"] + 
                    progress * (stage["lambda_end"] - stage["lambda_start"])
                )
                self.current_lambda = lambda_val
                self.history.append({
                    "epoch": epoch,
                    "lambda": lambda_val,
                    "stage": stage["name"],
                })
                return lambda_val
        
        # Beyond all stages: use last stage's end value
        last_stage = list(self.schedule.values())[-1]
        self.current_lambda = last_stage["lambda_end"]
        self.history.append({
            "epoch": epoch,
            "lambda": self.current_lambda,
            "stage": "post-schedule",
        })
        return self.current_lambda
    
    def get_current_stage(self, epoch: int) -> str:
        """Return the name of the current training stage."""
        for stage in self.schedule.values():
            if stage["start_epoch"] <= epoch < stage["end_epoch"]:
                return stage["name"]
        return "post-schedule"
    
    def get_schedule_summary(self) -> str:
        """Return a formatted summary of the sparsity schedule."""
        lines = ["Sparsity Schedule:"]
        for key, stage in self.schedule.items():
            lines.append(
                f"  {stage['name']:15s} | "
                f"Epochs {stage['start_epoch']:3d}-{stage['end_epoch']:3d} | "
                f"L: {stage['lambda_start']:.4f} -> {stage['lambda_end']:.4f}"
            )
        return "\n".join(lines)
