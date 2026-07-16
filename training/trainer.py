"""
AgroGrow Training Pipeline Module.
Implements the training, validation, early stopping, learning rate scheduling,
mixed precision training, and checkpointing for Corn-Net.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
from AgroGrow.config import global_config
from AgroGrow.utils.logger import logger
from AgroGrow.utils.metrics import MetricsEvaluator, plot_training_curves

class CornNetTrainer:
    """
    Orchestrates training and validation loops, checkpointing, and evaluation metrics plotting.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        
        self.scaler = GradScaler(enabled=(device.type == "cuda"))
        self.evaluator = MetricsEvaluator(num_classes=global_config.num_classes)
        
        # Output directory paths
        self.checkpoint_path = global_config.weights_dir / "checkpoint.pth"
        self.best_model_path = global_config.weights_dir / "best_model.pth"
        self.history_path = global_config.results_dir / "training_history.json"
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_miou": [],
            "val_miou": [],
            "train_dice": [],
            "val_dice": [],
            "pixel_acc": []
        }
        self.start_epoch = 0

    def train_epoch(self) -> Tuple[float, float, float]:
        """Runs a single epoch of training. Returns (mean_loss, mean_miou, mean_dice)."""
        self.model.train()
        total_loss = 0.0
        
        # Cumulative confusion matrix for training set
        epoch_cm = np.zeros((self.evaluator.num_classes, self.evaluator.num_classes), dtype=np.int64)
        
        for images, masks, _ in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Autocast for mixed precision
            with autocast(enabled=(self.device.type == "cuda")):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item() * images.size(0)
            
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            targets = masks.detach().cpu().numpy()
            epoch_cm += self.evaluator.compute_confusion_matrix(preds, targets)
            
        mean_loss = total_loss / len(self.train_loader.dataset)
        
        # Compute training metrics from confusion matrix
        metrics = self.evaluator.evaluate_from_confusion_matrix(epoch_cm)
        
        return mean_loss, metrics["mean_iou"], metrics["mean_dice"]

    def val_epoch(self) -> Tuple[float, float, float, float]:
        """Runs a validation pass. Returns (mean_loss, mean_miou, mean_dice, pixel_acc)."""
        self.model.eval()
        total_loss = 0.0
        
        # Cumulative confusion matrix for validation set
        epoch_cm = np.zeros((self.evaluator.num_classes, self.evaluator.num_classes), dtype=np.int64)
        
        with torch.no_grad():
            for images, masks, _ in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                
                with autocast(enabled=(self.device.type == "cuda")):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    
                total_loss += loss.item() * images.size(0)
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                targets = masks.cpu().numpy()
                epoch_cm += self.evaluator.compute_confusion_matrix(preds, targets)
                
        mean_loss = total_loss / len(self.val_loader.dataset)
        
        # Compute metrics from confusion matrix
        metrics = self.evaluator.evaluate_from_confusion_matrix(epoch_cm)
        
        return mean_loss, metrics["mean_iou"], metrics["mean_dice"], metrics["pixel_accuracy"]

    def fit(self, num_epochs: int = None):
        """Orchestrates fitting loop with early stopping, validation, checkpoints, and visualization."""
        epochs = num_epochs if num_epochs is not None else global_config.epochs
        best_val_loss = float("inf")
        epochs_no_improve = 0
        
        logger.info(f"Training started on device: {self.device}")
        
        for epoch in range(self.start_epoch, epochs):
            train_loss, train_miou, train_dice = self.train_epoch()
            val_loss, val_miou, val_dice, val_acc = self.val_epoch()
            
            # Step learning rate scheduler
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(val_loss)
            else:
                self.lr_scheduler.step()
                
            # Log progress
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] - "
                f"Train Loss: {train_loss:.4f}, mIoU: {train_miou:.4f}, Dice: {train_dice:.4f} | "
                f"Val Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}, Dice: {val_dice:.4f}, Acc: {val_acc:.4f}"
            )
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_miou"].append(train_miou)
            self.history["val_miou"].append(val_miou)
            self.history["train_dice"].append(train_dice)
            self.history["val_dice"].append(val_dice)
            self.history["pixel_acc"].append(val_acc)
            
            # Save regular checkpoint
            self.save_checkpoint(epoch)
            
            # Early stopping and best model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.save_best_model(val_loss, val_miou)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= global_config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                    break
                    
        # Generate metric plots
        plot_training_curves(self.history, global_config.results_dir)
        
        # Save history JSON
        with open(self.history_path, "w", encoding="utf-8") as hf:
            json.dump(self.history, hf, indent=4)
            
        logger.info("Training session completed successfully.")

    def save_checkpoint(self, epoch: int):
        """Saves current state training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "history": self.history
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self) -> bool:
        """Resumes training state from checkpoint.pth if it exists."""
        if not self.checkpoint_path.exists():
            logger.info("No checkpoint found to resume from.")
            return False
            
        logger.info(f"Resuming training from checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.history = checkpoint["history"]
        self.start_epoch = checkpoint["epoch"] + 1
        return True

    def save_best_model(self, loss: float, miou: float):
        """Saves model weights corresponding to the best validation loss."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "best_loss": loss,
            "best_miou": miou
        }, self.best_model_path)
        logger.info(f"New best model saved with Val Loss: {loss:.4f}, mIoU: {miou:.4f}")
