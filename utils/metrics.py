"""
AgroGrow Metrics and Visualization Module.
Provides functions to compute multi-class segmentation metrics:
Pixel Accuracy, IoU, Mean IoU, Dice/F1 Score, Precision, Recall,
and saves training performance curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from AgroGrow.config import global_config
from AgroGrow.utils.logger import logger

class MetricsEvaluator:
    """
    Computes confusion-matrix based semantic segmentation metrics.
    """
    def __init__(self, num_classes: int = 4, eps: float = 1e-7):
        self.num_classes = num_classes
        self.eps = eps

    def compute_confusion_matrix(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Computes the confusion matrix of size (num_classes, num_classes).
        Row indicates target, Column indicates prediction.
        """
        # Flatten arrays
        preds_flat = preds.reshape(-1)
        targets_flat = targets.reshape(-1)
        
        # Filter valid target labels
        mask = (targets_flat >= 0) & (targets_flat < self.num_classes)
        preds_flat = preds_flat[mask]
        targets_flat = targets_flat[mask]
        
        # Fast confusion matrix computation
        category = targets_flat * self.num_classes + preds_flat
        counts = np.bincount(category, minlength=self.num_classes**2)
        confusion_matrix = counts.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def evaluate_from_confusion_matrix(self, cm: np.ndarray) -> Dict[str, any]:
        """
        Calculates all semantic metrics directly from a cumulative confusion matrix.
        """
        # Pixel Accuracy: sum of diagonal / sum of all
        sum_diagonal = np.diag(cm).sum()
        total_pixels = cm.sum()
        pixel_accuracy = sum_diagonal / (total_pixels + self.eps)
        
        # Class-wise metrics
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        
        iou = tp / (tp + fp + fn + self.eps)
        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + self.eps)
        
        return {
            "pixel_accuracy": float(pixel_accuracy),
            "class_iou": iou.tolist(),
            "mean_iou": float(np.mean(iou)),
            "class_precision": precision.tolist(),
            "mean_precision": float(np.mean(precision)),
            "class_recall": recall.tolist(),
            "mean_recall": float(np.mean(recall)),
            "class_dice": dice.tolist(),
            "mean_dice": float(np.mean(dice)),
            "class_f1": dice.tolist(),  # F1 and Dice are equivalent
            "mean_f1": float(np.mean(dice))
        }

    def evaluate(self, preds: np.ndarray, targets: np.ndarray) -> Dict[str, any]:
        """
        Evaluates metrics from prediction map and target map.
        
        Args:
            preds (np.ndarray): Integers, shape (H, W) or (N, H, W).
            targets (np.ndarray): Integers, shape (H, W) or (N, H, W).
            
        Returns:
            Dict[str, any]: Contains pixel_accuracy, class_iou, mean_iou,
                            class_dice, mean_dice, class_precision, mean_precision,
                            class_recall, mean_recall, class_f1, mean_f1.
        """
        cm = self.compute_confusion_matrix(preds, targets)
        return self.evaluate_from_confusion_matrix(cm)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_dir: Path
):
    """
    Plots training and validation curves for loss and performance metrics.
    
    Args:
        history (Dict[str, List[float]]): Dict with keys like 'train_loss', 'val_loss',
                                          'train_miou', 'val_miou', etc.
        save_dir (Path): Folder to save the plotted image.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(18, 5))
    
    # 1. Loss Plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    plt.title("Loss Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    
    # 2. Mean IoU Plot
    plt.subplot(1, 3, 2)
    if "train_miou" in history and "val_miou" in history:
        plt.plot(epochs, history["train_miou"], "b-", label="Train mIoU", linewidth=2)
        plt.plot(epochs, history["val_miou"], "r-", label="Val mIoU", linewidth=2)
    plt.title("Mean IoU Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("mIoU", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    
    # 3. Dice Score Plot
    plt.subplot(1, 3, 3)
    if "train_dice" in history and "val_dice" in history:
        plt.plot(epochs, history["train_dice"], "b-", label="Train Dice", linewidth=2)
        plt.plot(epochs, history["val_dice"], "r-", label="Val Dice", linewidth=2)
    plt.title("Mean Dice Score Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Dice Score", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plot_path = save_dir / "training_metrics_curves.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved metric curves to {plot_path}")
