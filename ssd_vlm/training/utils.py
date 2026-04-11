"""Training utilities for SSD-VLM."""

import logging
import math
from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


class CosineWarmupScheduler(LambdaLR):
    """Cosine annealing with linear warmup."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            num_cycles: Number of cosine cycles
            last_epoch: Last epoch (for resuming)
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.num_cycles = num_cycles
        
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        
        super().__init__(optimizer, lr_lambda, last_epoch)


class GradualWarmupScheduler(LambdaLR):
    """Linear learning rate warmup."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            last_epoch: Last epoch (for resuming)
        """
        self.warmup_steps = warmup_steps
        
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
        
        super().__init__(optimizer, lr_lambda, last_epoch)


def freeze_model(model: nn.Module, freeze: bool = True):
    """Freeze model parameters."""
    for param in model.parameters():
        param.requires_grad = not freeze


def unfreeze_model(model: nn.Module):
    """Unfreeze all model parameters."""
    freeze_model(model, freeze=False)


def freeze_vision_encoder(model: nn.Module):
    """Freeze vision encoder (for LoRA training)."""
    # This is model-specific - adapt to your architecture
    if hasattr(model, "vision_tower"):
        freeze_model(model.vision_tower, freeze=True)
    if hasattr(model, "image_processor"):
        freeze_model(model.image_processor, freeze=True)


def log_model_info(model: nn.Module):
    """Log model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
    logger.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")


def get_model_memory(model: nn.Module, dtype: torch.dtype = torch.float32) -> float:
    """
    Estimate model memory in GB.
    
    Args:
        model: PyTorch model
        dtype: Data type
    
    Returns:
        Memory in GB
    """
    dtype_size = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }
    
    bytes_per_param = dtype_size.get(dtype, 4)
    total_params = sum(p.numel() for p in model.parameters())
    memory_gb = (total_params * bytes_per_param) / (1024 ** 3)
    
    return memory_gb


def log_gradient_stats(model: nn.Module, step: int, logger_obj=None):
    """Log gradient statistics."""
    if logger_obj is None:
        logger_obj = logger
    
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    logger_obj.info(f"Step {step}: Gradient norm = {total_norm:.4f}")


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[Optimizer],
    scheduler: Optional[object],
    epoch: int,
    step: int,
    output_dir: str,
):
    """Save training checkpoint."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}_step{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    return checkpoint_path


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: Optional[Optimizer] = None):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    
    logger.info(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch}, step {step})")
    
    return epoch, step
