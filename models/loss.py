"""
AgroGrow Segmentation Loss Functions Module.
Implements Weighted Cross Entropy, Multi-Class Dice Loss,
and their Hybrid combination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class DiceLoss(nn.Module):
    """
    Multi-class Soft Dice Loss.
    Computes overlap score and returns 1 - Dice.
    """
    def __init__(self, smooth: float = 1e-6, ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Model raw predictions, shape (B, C, H, W).
            targets (torch.Tensor): Ground truth indices, shape (B, H, W).
            
        Returns:
            torch.Tensor: Scalar Dice loss value.
        """
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        
        # Convert targets to one-hot: shape (B, C, H, W)
        targets_one_hot = F.one_hot(targets.clamp(min=0), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Handle ignore_index if active
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).float().unsqueeze(1)
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask
            
        # Compute intersection and sums over batch, spatial dimensions (axes 0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=(0, 2, 3))
        cardinality_probs = torch.sum(probs * probs, dim=(0, 2, 3))  # squared values for smooth optimization
        cardinality_targets = torch.sum(targets_one_hot * targets_one_hot, dim=(0, 2, 3))
        
        dice_coeffs = (2.0 * intersection + self.smooth) / (cardinality_probs + cardinality_targets + self.smooth)
        
        # Return average Dice Loss across all classes (excluding background or averaging all)
        return 1.0 - torch.mean(dice_coeffs)

class HybridLoss(nn.Module):
    """
    Hybrid Loss: Combined Weighted Cross Entropy and Multi-Class Dice Loss.
    """
    def __init__(self, class_weights: List[float] = None, wce_weight: float = 0.5, dice_weight: float = 0.5):
        """
        Args:
            class_weights (List[float]): Weights for each class in CrossEntropy.
            wce_weight (float): Multiplier for Weighted Cross Entropy.
            dice_weight (float): Multiplier for Dice Loss.
        """
        super().__init__()
        self.wce_weight = wce_weight
        self.dice_weight = dice_weight
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        else:
            weights_tensor = None
            
        self.wce = nn.CrossEntropyLoss(weight=weights_tensor)
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.wce_weight > 0:
            loss += self.wce_weight * self.wce(logits, targets)
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice(logits, targets)
        return loss
