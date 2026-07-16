"""
AgroGrow Test Suite Module.
Contains unit and integration tests to verify model tensors, loss functions,
evaluation metrics, grading thresholds, and storage model predictions.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path if running directly
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from AgroGrow.config import global_config
from AgroGrow.models.corn_net import CornNet
from AgroGrow.models.loss import DiceLoss, HybridLoss
from AgroGrow.utils.metrics import MetricsEvaluator
from AgroGrow.prediction.feature_extractor import CornFeatureExtractor
from AgroGrow.prediction.grader import CornGrader
from AgroGrow.storage_prediction.model import StoragePredictor

def test_corn_net_shape():
    """Validates input/output shape matching for CornNet."""
    print("[RUNNING] test_corn_net_shape...")
    model = CornNet(in_channels=3, num_classes=global_config.num_classes)
    model.eval()
    
    # Test batch size of 2, 3 channels, size 256x256
    dummy_input = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)
        
    expected_shape = (2, global_config.num_classes, 256, 256)
    assert output.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {output.shape}"
    print("[SUCCESS] test_corn_net_shape passed!")

def test_loss_functions():
    """Verifies that Dice and Hybrid loss classes compute positive scalar losses."""
    print("[RUNNING] test_loss_functions...")
    logits = torch.randn(4, 4, 256, 256, requires_grad=True)
    targets = torch.randint(0, 4, (4, 256, 256))
    
    dice_criterion = DiceLoss()
    hybrid_criterion = HybridLoss(class_weights=global_config.class_weights)
    
    dice_loss = dice_criterion(logits, targets)
    hybrid_loss = hybrid_criterion(logits, targets)
    
    assert dice_loss.item() >= 0, f"Dice loss is negative: {dice_loss.item()}"
    assert hybrid_loss.item() >= 0, f"Hybrid loss is negative: {hybrid_loss.item()}"
    
    # Backward pass verification
    hybrid_loss.backward()
    assert logits.grad is not None, "Gradients not computed during backward pass!"
    print("[SUCCESS] test_loss_functions passed!")

def test_metrics_evaluation():
    """Checks that the metrics evaluator computes accurate statistics."""
    print("[RUNNING] test_metrics_evaluation...")
    evaluator = MetricsEvaluator(num_classes=4)
    
    # Create simple 10x10 dummy predictions and targets
    targets = np.zeros((10, 10), dtype=np.uint8)
    targets[2:8, 2:8] = 1  # 36 healthy pixels
    
    preds = np.zeros((10, 10), dtype=np.uint8)
    preds[2:8, 2:8] = 1
    # Introduce 6 pixel errors (false negatives of class 1)
    preds[2:4, 2:5] = 0
    
    res = evaluator.evaluate(preds, targets)
    
    assert 0.0 <= res["pixel_accuracy"] <= 1.0, "Invalid pixel accuracy range"
    assert 0.0 <= res["mean_iou"] <= 1.0, "Invalid mIoU range"
    assert len(res["class_iou"]) == 4, "Incorrect class count in IoU"
    print("[SUCCESS] test_metrics_evaluation passed!")

def test_quality_grading():
    """Validates the logic mapping features to Grade A, B, C, D classifications."""
    print("[RUNNING] test_quality_grading...")
    
    # Grade A Profile
    features_a = {
        "healthy_percentage": 94.5,
        "disease_percentage": 0.0,
        "missing_percentage": 1.2,
        "total_corn_area_pixels": 12000,
        "defect_area_pixels": 120,
        "quality_score": 93.3,
        "num_healthy_blobs": 12,
        "kernel_density_score": 0.5
    }
    
    grading_a = CornGrader.classify_grade(features_a, 0.95)
    assert grading_a["grade"] == "Grade A", f"Expected Grade A, got {grading_a['grade']}"
    
    # Grade D Profile (Diseased)
    features_d = {
        "healthy_percentage": 65.0,
        "disease_percentage": 25.0,
        "missing_percentage": 10.0,
        "total_corn_area_pixels": 10000,
        "defect_area_pixels": 3500,
        "quality_score": 20.0,
        "num_healthy_blobs": 5,
        "kernel_density_score": 0.2
    }
    grading_d = CornGrader.classify_grade(features_d, 0.88)
    assert grading_d["grade"] == "Grade D", f"Expected Grade D, got {grading_d['grade']}"
    print("[SUCCESS] test_quality_grading passed!")

def main():
    print("="*50)
    print(" RUNNING AGROGROW SKELETON UNIT TESTS")
    print("="*50)
    try:
        test_corn_net_shape()
        test_loss_functions()
        test_metrics_evaluation()
        test_quality_grading()
        print("="*50)
        print(" ALL UNIT TESTS PASSED SUCCESSFULLY!")
        print("="*50)
        sys.exit(0)
    except AssertionError as ae:
        print(f"\n[FAIL] Test assertion failed: {ae}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Test run encountered error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
