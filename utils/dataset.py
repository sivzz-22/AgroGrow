"""
AgroGrow Dataset Preprocessing Module.
Implements the PyTorch Dataset, Albumentations augmentations, DataLoaders,
and visualization utilities like color overlay generation.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import Tuple, List
from AgroGrow.config import global_config
from AgroGrow.utils.logger import logger

class CornDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Corn images and segmentation masks.
    """
    def __init__(self, split: str, augment: bool = False):
        """
        Args:
            split (str): One of "train", "val", "test".
            augment (bool): Whether to apply data augmentation.
        """
        self.split = split
        self.augment = augment
        self.images_dir = global_config.images_dir / split
        self.masks_dir = global_config.masks_dir / split
        
        # Gather all image files
        self.image_paths = sorted(list(self.images_dir.glob("*.jpg")))
        
        if len(self.image_paths) == 0:
            logger.warning(f"No images found for split '{split}' in {self.images_dir}")
            
        # Define transform pipeline
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        # Base transformation (Resize and Normalize)
        base_transforms = [
            A.Resize(height=global_config.input_size[0], width=global_config.input_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]
        
        if self.augment:
            # Training augmentations
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                *base_transforms
            ])
        else:
            self.transform = A.Compose(base_transforms)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Loads image and mask, applies transforms, and returns:
            - Normalized image tensor: shape (3, H, W)
            - Target mask tensor: shape (H, W)
            - Filename string (useful for validation tracking)
        """
        img_path = self.image_paths[idx]
        mask_path = self.masks_dir / f"{img_path.stem}.png"
        
        # Read image (BGR -> RGB)
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Read mask (Grayscale)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply transformations
        transformed = self.transform(image=img, mask=mask)
        img_tensor = transformed["image"]
        mask_tensor = transformed["mask"].long()  # Must be long tensor for Cross Entropy Loss
        
        return img_tensor, mask_tensor, img_path.name

def get_data_loaders(
    batch_size: int = None,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for Train, Val, and Test splits.
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    bs = batch_size if batch_size is not None else global_config.batch_size
    
    train_dataset = CornDataset(split="train", augment=True)
    val_dataset = CornDataset(split="val", augment=False)
    test_dataset = CornDataset(split="test", augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    logger.info(f"Initialized DataLoaders. Train batches: {len(train_loader)}, "
                f"Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
                
    return train_loader, val_loader, test_loader

def generate_overlay(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Generates a color overlay of the segmentation mask on top of the original image.
    
    Args:
        image_rgb (np.ndarray): Original image in RGB format, shape (H, W, 3).
        mask (np.ndarray): Segmentation mask, shape (H, W), values 0 to num_classes-1.
        alpha (float): Transparency parameter for blending.
        
    Returns:
        np.ndarray: Blended RGB image.
    """
    # Create empty color canvas
    color_mask = np.zeros_like(image_rgb)
    colors = global_config.class_colors
    
    for class_idx, color in enumerate(colors):
        color_mask[mask == class_idx] = color
        
    # Blend color mask and original image
    overlay = cv2.addWeighted(image_rgb, 1.0, color_mask, alpha, 0)
    return overlay
