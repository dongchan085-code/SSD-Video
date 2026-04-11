"""
LoRA Fine-tuning for SSD-VLM (Stage 1).
Fine-tune with LoRA adapters (rank 128, alpha 256) on SSD-generated samples.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    get_scheduler,
)

from ssd_vlm.data.ssd_sample_dataset import (
    SSDSampleDataset,
    SSDSampleDataCollator,
    create_ssd_sample_dataloader,
)
from ssd_vlm.training.utils import (
    CosineWarmupScheduler,
    log_model_info,
    log_gradient_stats,
    save_checkpoint,
)

logger = logging.getLogger(__name__)


class LoRATrainer:
    """Trainer for LoRA fine-tuning on SSD samples."""
    
    def __init__(
        self,
        model_id: str,
        output_dir: str,
        lora_config: Dict[str, Any],
        training_config: Dict[str, Any],
        device: str = "cuda",
    ):
        """
        Initialize LoRA trainer.
        
        Args:
            model_id: HuggingFace model ID
            output_dir: Output directory for checkpoints
            lora_config: LoRA configuration
            training_config: Training hyperparameters
            device: Device to train on
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.lora_config = lora_config
        self.training_config = training_config
        
        # Load model and processor
        logger.info(f"Loading model: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Apply LoRA
        logger.info("Applying LoRA configuration")
        lora = LoraConfig(
            r=lora_config.get("r", 128),
            lora_alpha=lora_config.get("lora_alpha", 256),
            lora_dropout=lora_config.get("lora_dropout", 0.1),
            bias=lora_config.get("bias", "none"),
            task_type=lora_config.get("task_type", "CAUSAL_LM"),
            target_modules=lora_config.get("target_modules", [
                "q_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"
            ]),
            modules_to_save=lora_config.get("modules_to_save"),
            init_lora_weights=lora_config.get("init_lora_weights", True),
        )
        
        self.model = get_peft_model(self.model, lora)
        self.model.print_trainable_parameters()
        
        log_model_info(self.model)
        
        # Setup optimizer and scheduler
        self.num_epochs = training_config.get("num_train_epochs", 3)
        self.learning_rate = training_config.get("learning_rate", 5e-4)
        self.warmup_ratio = training_config.get("warmup_ratio", 0.1)
        
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
        Train LoRA on SSD samples.
        
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
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.get("max_grad_norm", 1.0)
                    )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    epoch_loss += loss.item()
                    global_step += 1
                    
                    # Logging
                    if global_step % self.training_config.get("logging_steps", 10) == 0:
                        avg_loss = epoch_loss / (batch_idx + 1)
                        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})
                        logger.info(f"Step {global_step}: Loss = {avg_loss:.4f}")
                    
                    # Save checkpoint
                    if global_step % self.training_config.get("save_steps", 100) == 0:
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


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for SSD-VLM")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--samples_path", type=str, required=True, help="Path to SSD samples JSONL")
    parser.add_argument("--output_dir", type=str, default="./outputs/lora_checkpoint",
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
    
    # Create trainer
    trainer = LoRATrainer(
        model_id=config["model"].get("model_id", "Qwen/Qwen3-VL-8B-Instruct"),
        output_dir=args.output_dir,
        lora_config=config["lora"],
        training_config=config["training"],
    )
    
    # Load data
    logger.info(f"Loading SSD samples from {args.samples_path}")
    train_dataloader = create_ssd_sample_dataloader(
        samples_path=args.samples_path,
        tokenizer=trainer.processor.tokenizer,
        batch_size=config["training"].get("per_device_train_batch_size", 2),
        num_workers=config["training"].get("dataloader_num_workers", 4),
        shuffle=True,
        max_seq_length=config["data"].get("max_seq_length", 4096),
    )
    
    # Train
    trainer.train(train_dataloader)


if __name__ == "__main__":
    main()
