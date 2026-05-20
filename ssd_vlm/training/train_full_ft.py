"""
Full-parameter Fine-tuning for SSD-VLM (Ablation Only).
This is NOT part of the main method — the paper uses LoRA-only.
Available as an ablation baseline via configs/ablations/ablation_full_ft_only.yaml.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    get_scheduler,
)

from ssd_vlm.data.ssd_sample_dataset import create_ssd_sample_dataloader
from ssd_vlm.training.utils import (
    log_model_info,
    log_gradient_stats,
    save_checkpoint,
)
from ssd_vlm.utils.config import load_config
from ssd_vlm.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


class FullFTTrainer:
    """Trainer for full-parameter fine-tuning on SSD samples (Stage 2)."""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        training_config: Dict[str, Any],
        device: str = "cuda",
    ):
        """
        Initialize full-parameter fine-tuning trainer.
        
        Args:
            model_path: Path to merged LoRA checkpoint or base model
            output_dir: Output directory for checkpoints
            training_config: Training hyperparameters
            device: Device to train on
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.training_config = training_config
        
        # Load model and processor
        logger.info(f"Loading model from: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Ensure all parameters are trainable for full FT
        for param in self.model.parameters():
            param.requires_grad = True
        
        log_model_info(self.model)
        
        # Setup optimizer and scheduler
        self.num_epochs = training_config.get("num_train_epochs", 1)
        self.learning_rate = training_config.get("learning_rate", 2e-5)
        self.warmup_ratio = training_config.get("warmup_ratio", 0.05)
        
        self.optimizer = None
        self.scheduler = None
    
    def setup_optimizer(self, num_training_steps: int):
        """Setup optimizer and scheduler."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.training_config.get("weight_decay", 0.01),
        )
        
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        
        self.scheduler = get_scheduler(
            name=self.training_config.get("lr_scheduler_type", "cosine"),
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        logger.info(f"Optimizer setup: lr={self.learning_rate}, warmup_steps={num_warmup_steps}")
    
    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """
        Train with full parameters on SSD samples.
        
        Args:
            train_dataloader: Training dataloader
            eval_dataloader: Optional evaluation dataloader
        """
        num_training_steps = len(train_dataloader) * self.num_epochs
        self.setup_optimizer(num_training_steps)
        
        self.model.to(self.device)
        
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Training loop
            self.model.train()
            epoch_loss = 0.0
            
            with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}") as pbar:
                for batch_idx, batch in enumerate(train_dataloader):
                    # Move batch to device
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    
                    loss = outputs.loss
                    
                    # Backward pass with gradient accumulation
                    grad_accum_steps = self.training_config.get("gradient_accumulation_steps", 16)
                    loss = loss / grad_accum_steps
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.get("max_grad_norm", 1.0)
                    )
                    
                    # Update on gradient accumulation boundary
                    if (batch_idx + 1) % grad_accum_steps == 0:
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        global_step += 1
                    
                    epoch_loss += loss.item() * grad_accum_steps
                    
                    # Logging
                    if global_step % self.training_config.get("logging_steps", 10) == 0:
                        avg_loss = epoch_loss / (batch_idx + 1)
                        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})
                        logger.info(f"Step {global_step}: Loss = {avg_loss:.4f}")
                    
                    # Save checkpoint
                    if global_step % self.training_config.get("save_steps", 50) == 0:
                        save_checkpoint(
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            epoch,
                            global_step,
                            str(self.output_dir / "checkpoints"),
                        )
                    
                    pbar.update(1)
            
            # Evaluation
            if eval_dataloader is not None:
                eval_loss = self._evaluate(eval_dataloader)
                logger.info(f"Epoch {epoch + 1} - Eval Loss: {eval_loss:.4f}")
                
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    self._save_model(epoch, "best")
            
            # Save epoch checkpoint
            self._save_model(epoch, suffix="")
        
        logger.info("Training complete!")
        self._save_model(-1, "final")
    
    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
        
        self.model.train()
        
        return total_loss / num_batches
    
    def _save_model(self, epoch: int, suffix: str = ""):
        """Save model checkpoint."""
        if suffix:
            save_dir = self.output_dir / f"checkpoint_epoch{epoch}_{suffix}"
        else:
            save_dir = self.output_dir / f"checkpoint_epoch{epoch}"
        
        self.model.save_pretrained(str(save_dir))
        self.processor.save_pretrained(str(save_dir))
        
        logger.info(f"Model saved to {save_dir}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Full-parameter fine-tuning for SSD-VLM")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--samples_path", type=str, required=True, help="Path to SSD samples JSONL")
    parser.add_argument("--lora_checkpoint", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output_dir", type=str, default="./outputs/ssd_vlm_final",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    set_global_seed(int(config.get("seed", 42)))
    
    # Determine model path (merged LoRA or base model)
    model_path = args.lora_checkpoint
    if config["checkpoint"].get("merge_lora_before_training", True):
        logger.info("Note: LoRA checkpoint should be pre-merged before full FT")
    
    # Create trainer
    trainer = FullFTTrainer(
        model_path=model_path,
        output_dir=args.output_dir,
        training_config=config["training"],
    )
    
    # Load data
    logger.info(f"Loading SSD samples from {args.samples_path}")
    train_dataloader = create_ssd_sample_dataloader(
        samples_path=args.samples_path,
        tokenizer=trainer.processor.tokenizer,
        batch_size=config["training"].get("per_device_train_batch_size", 1),
        num_workers=config["training"].get("dataloader_num_workers", 4),
        shuffle=True,
        max_seq_length=config["data"].get("max_seq_length", 4096),
    )
    
    # Train
    trainer.train(train_dataloader)


if __name__ == "__main__":
    main()
