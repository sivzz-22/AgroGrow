"""
AgroGrow Training Pipeline Entrypoint.
Runs the dataset setup, initializes model and metrics, and launches the trainer.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from AgroGrow.config import global_config
from AgroGrow.utils.logger import logger
from AgroGrow.utils.prepare_dataset import DatasetPreparer
from AgroGrow.utils.validation import DatasetValidator
from AgroGrow.utils.dataset import get_data_loaders
from AgroGrow.models.corn_net import CornNet
from AgroGrow.models.loss import HybridLoss
from AgroGrow.training.trainer import CornNetTrainer

def main():
    logger.info("Initializing AgroGrow Deep Learning Training Pipeline...")
    
    # 1. Dataset Check & Auto-Preparation
    train_img_dir = global_config.images_dir / "train"
    if not train_img_dir.exists() or len(list(train_img_dir.glob("*.jpg"))) == 0:
        logger.info("Prepared dataset not found. Initializing auto-preprocessing and validation...")
        
        preparer = DatasetPreparer(
            raw_dir=global_config.raw_dataset_dir,
            target_dir=global_config.dataset_dir
        )
        preparer.process()
        
        validator = DatasetValidator(dataset_dir=global_config.dataset_dir)
        success, msg = validator.validate()
        if not success:
            logger.critical(f"Dataset auto-preparation failed: {msg}")
            sys.exit(1)
            
    # 2. Setup Device (GPU support with CPU fallback)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    logger.info(f"Using execution device: {device_name}")
    
    if device_name == "cuda":
        # Log active GPU info
        logger.info(f"Active GPU Device: {torch.cuda.get_device_name(0)}")
        # Enable benchmark mode for faster training of fixed-input shapes
        torch.backends.cudnn.benchmark = True

    # 3. Initialize DataLoaders
    train_loader, val_loader, _ = get_data_loaders()

    # 4. Initialize Network, Optimizer, Loss, and Scheduler
    model = CornNet(num_classes=global_config.num_classes)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=global_config.learning_rate,
        weight_decay=global_config.weight_decay
    )
    
    criterion = HybridLoss(
        class_weights=global_config.class_weights,
        wce_weight=global_config.wce_weight,
        dice_weight=global_config.dice_weight
    )
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=global_config.lr_scheduler_factor,
        patience=global_config.lr_scheduler_patience
    )

    # 5. Initialize Trainer and run
    trainer = CornNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device
    )
    
    # Check if a checkpoint exists to resume training
    if trainer.load_checkpoint():
        logger.info("Resuming training from loaded checkpoint.")
        
    trainer.fit()
    
    logger.info("Training pipeline execution finished.")

if __name__ == "__main__":
    main()
