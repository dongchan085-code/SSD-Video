"""
LoRA Fine-tuning for SSD-VLM (Stage 1).
Fine-tune with LoRA adapters (rank 128, alpha 256) on SSD-generated samples.
"""

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    get_scheduler,
)

from ssd_vlm.data.ssd_sample_dataset import (
    create_ssd_sample_dataloaders,
)
from ssd_vlm.training.utils import (
    log_model_info,
    save_checkpoint,
)
from ssd_vlm.utils.config import load_config

try:
    from accelerate import Accelerator
except ImportError:  # pragma: no cover - optional at import time
    Accelerator = None

logger = logging.getLogger(__name__)


class LoRATrainer:
    """Trainer for LoRA fine-tuning on SSD samples."""
    
    def __init__(
        self,
        model_id: str,
        output_dir: str,
        lora_config: Dict[str, Any],
        training_config: Dict[str, Any],
        model_config: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
    ):
        """
        Initialize LoRA trainer.

        Args:
            model_id: HuggingFace model ID
            output_dir: Output directory for checkpoints
            lora_config: LoRA configuration
            training_config: Training hyperparameters
            model_config: Model loading config (dtype, device_map)
            device: Device to train on
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        self.lora_config = lora_config
        self.training_config = training_config
        self.gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
        backend = training_config.get("backend", "accelerate")
        self.accelerator = (
            Accelerator(gradient_accumulation_steps=self.gradient_accumulation_steps)
            if backend == "accelerate" and Accelerator is not None
            else None
        )
        if self.accelerator is not None:
            self.device = self.accelerator.device

        # Resolve dtype and device_map from model config
        model_config = model_config or {}
        dtype_str = model_config.get("dtype", "bfloat16")
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                     "float32": torch.float32}
        torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)
        device_map = model_config.get("device_map", "auto")
        if self.accelerator is not None:
            device_map = None

        # Load model and processor
        logger.info(f"Loading model: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
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
        self.num_epochs = training_config.get("num_train_epochs", 2)
        self.learning_rate = training_config.get("learning_rate", 5e-4)
        self.warmup_ratio = training_config.get("warmup_ratio", 0.1)
        self.early_stopping_patience = training_config.get("early_stopping_patience", 3)

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
        num_training_steps = max(
            1,
            math.ceil(len(train_dataloader) / self.gradient_accumulation_steps) * self.num_epochs,
        )
        self.setup_optimizer(num_training_steps)

        if self.accelerator is not None:
            if eval_dataloader is not None:
                self.model, self.optimizer, train_dataloader, eval_dataloader, self.scheduler = (
                    self.accelerator.prepare(
                        self.model,
                        self.optimizer,
                        train_dataloader,
                        eval_dataloader,
                        self.scheduler,
                    )
                )
            else:
                self.model, self.optimizer, train_dataloader, self.scheduler = (
                    self.accelerator.prepare(
                        self.model,
                        self.optimizer,
                        train_dataloader,
                        self.scheduler,
                    )
                )
        elif not getattr(self.model, "hf_device_map", None):
            self.model.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        
        optimizer_step = 0
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Training loop
            self.model.train()
            epoch_loss = 0.0
            
            with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}") as pbar:
                for batch_idx, batch in enumerate(train_dataloader):
                    # Move batch to device
                    if self.accelerator is None:
                        batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                                for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    
                    loss = outputs.loss
                    
                    # Backward pass
                    scaled_loss = loss / self.gradient_accumulation_steps
                    if self.accelerator is not None:
                        self.accelerator.backward(scaled_loss)
                    else:
                        scaled_loss.backward()
                    
                    epoch_loss += loss.item()
                    should_step = (
                        (batch_idx + 1) % self.gradient_accumulation_steps == 0
                        or (batch_idx + 1) == len(train_dataloader)
                    )

                    if should_step:
                        if self.accelerator is not None:
                            self.accelerator.clip_grad_norm_(
                                self.model.parameters(),
                                self.training_config.get("max_grad_norm", 1.0),
                            )
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.training_config.get("max_grad_norm", 1.0)
                            )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        optimizer_step += 1
                    
                    # Logging
                    if should_step and optimizer_step % self.training_config.get("logging_steps", 10) == 0:
                        avg_loss = epoch_loss / (batch_idx + 1)
                        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})
                        logger.info(f"Step {optimizer_step}: Loss = {avg_loss:.4f}")
                    
                    # Save checkpoint
                    if should_step and optimizer_step % self.training_config.get("save_steps", 100) == 0:
                        if self.accelerator is not None and not self.accelerator.is_main_process:
                            pbar.update(1)
                            continue
                        save_checkpoint(
                            self._unwrap_model(),
                            self.optimizer,
                            self.scheduler,
                            epoch,
                            optimizer_step,
                            str(self.output_dir / "checkpoints"),
                        )
                    
                    pbar.update(1)
            
            # Evaluation + early stopping
            if eval_dataloader is not None:
                eval_loss = self._evaluate(eval_dataloader)
                logger.info(f"Epoch {epoch + 1} - Eval Loss: {eval_loss:.4f}")

                if eval_loss < best_loss:
                    best_loss = eval_loss
                    patience_counter = 0
                    self._save_model(epoch, "best")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement for {patience_counter}/{self.early_stopping_patience} evals")
                    if patience_counter >= self.early_stopping_patience:
                        logger.info("Early stopping triggered")
                        break

            # Save epoch checkpoint
            self._save_model(epoch, suffix="")
        
        logger.info("Training complete!")
        self._save_model(-1, "final")

        # Merge LoRA weights into base model for inference
        if self.accelerator is None or self.accelerator.is_main_process:
            logger.info("Merging LoRA weights into base model...")
            merged_model = self._unwrap_model().merge_and_unload()
            merged_dir = self.output_dir / "merged"
            self.processor.save_pretrained(str(merged_dir))
            merged_model.save_pretrained(str(merged_dir))
            logger.info(f"Merged model saved to {merged_dir}")
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
    
    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            if self.accelerator is None:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}
            
            outputs = self.model(**batch)
            
            total_loss += outputs.loss.item()
            num_batches += 1
        
        self.model.train()
        
        return total_loss / num_batches

    def _unwrap_model(self):
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(self.model)
        return self.model
    
    def _save_model(self, epoch: int, suffix: str = ""):
        """Save model checkpoint."""
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return
        if suffix:
            save_dir = self.output_dir / f"checkpoint_epoch{epoch}_{suffix}"
        else:
            save_dir = self.output_dir / f"checkpoint_epoch{epoch}"
        
        self._unwrap_model().save_pretrained(str(save_dir))
        self.processor.save_pretrained(str(save_dir))
        
        logger.info(f"Model saved to {save_dir}")


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
    
    # Resolve device from model config
    device_map = config.get("model", {}).get("device_map", "auto")
    if device_map == "cpu":
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Create trainer
    trainer = LoRATrainer(
        model_id=config["model"].get("model_id", "Qwen/Qwen3-VL-8B-Instruct"),
        output_dir=args.output_dir,
        lora_config=config["lora"],
        training_config=config["training"],
        model_config=config.get("model"),
        device=device,
    )
    
    # Load data
    logger.info(f"Loading SSD samples from {args.samples_path}")
    train_dataloader, eval_dataloader = create_ssd_sample_dataloaders(
        samples_path=args.samples_path,
        processor=trainer.processor,
        source_data_path=config["data"].get("source_data_path"),
        batch_size=config["training"].get("per_device_train_batch_size", 2),
        eval_batch_size=config["training"].get("per_device_eval_batch_size", 4),
        num_workers=config["training"].get("dataloader_num_workers", 4),
        num_frames=config["data"].get("num_frames", 4),
        frame_sampling_strategy=config["data"].get("frame_sampling_strategy", "uniform"),
        resize_shortest_edge=config["data"].get("resize_shortest_edge", 224),
        max_seq_length=config["data"].get("max_seq_length", 4096),
        vqa_buffer_ratio=config["data"].get("vqa_buffer_ratio", 0.1),
        validation_split_ratio=config["data"].get("validation_split_ratio", 0.0),
        pin_memory=config["training"].get("dataloader_pin_memory", True),
        drop_last=config["training"].get("dataloader_drop_last", False),
        seed=config.get("seed", 42),
        persistent_workers=config["training"].get("dataloader_persistent_workers", False),
        prefetch_factor=config["training"].get("dataloader_prefetch_factor", 2),
    )
    
    # Train
    trainer.train(train_dataloader, eval_dataloader=eval_dataloader)


if __name__ == "__main__":
    main()
