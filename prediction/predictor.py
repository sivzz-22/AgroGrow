"""
AgroGrow Segmentation Prediction Module.
Handles loading trained model weights and running inference on single images,
batches, or folders, returning segmentations, overlays, and probability maps.
"""

import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Union
from AgroGrow.config import global_config
from AgroGrow.models.corn_net import CornNet
from AgroGrow.utils.logger import logger
from AgroGrow.utils.dataset import generate_overlay

class CornPredictor:
    """
    Loads weights and performs model inference.
    """
    def __init__(self, weights_path: Path = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CornNet(num_classes=global_config.num_classes)
        
        path = weights_path if weights_path is not None else global_config.weights_dir / "best_model.pth"
        
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded predictor model weights from: {path}")
        else:
            logger.warning(f"Model weights file not found at {path}. Model initialized with random weights.")
            
        self.model.to(self.device)
        self.model.eval()

        # Input normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, img_rgb: np.ndarray) -> torch.Tensor:
        """Resizes, normalizes, and converts image to PyTorch tensor."""
        img_resized = cv2.resize(img_rgb, global_config.input_size)
        img_norm = (img_resized.astype(np.float32) / 255.0 - self.mean) / self.std
        tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def refine_mask_with_cv(self, img_rgb: np.ndarray, model_mask: np.ndarray) -> np.ndarray:
        """
        Refines the model-predicted mask using CV heuristics:
        1. Finds the convex hull of predicted corn regions to identify the full cob boundaries.
        2. Inspects pixels inside the cob boundary that deviate from healthy yellow/green
           and classifies them as diseased/missing (representing mold/rot).
        """
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        refined_mask = model_mask.copy()
        
        # Create a binary mask of predicted corn regions
        binary = (model_mask > 0).astype(np.uint8)
        
        # Close small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of predicted cobs
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cob_hull = np.zeros_like(binary)
        for cnt in contours:
            if cv2.contourArea(cnt) > 2000:  # Ignore small background noises
                hull = cv2.convexHull(cnt)
                cv2.drawContours(cob_hull, [hull], -1, 1, -1)
                
        # Loop over pixels inside the cob hull to check for defect anomalies
        h, w = model_mask.shape
        for y in range(h):
            for x in range(w):
                if cob_hull[y, x] > 0:
                    h_val, s_val, v_val = hsv[y, x]
                    
                    # Check if the pixel has healthy yellow/green colors
                    is_yellow = (10 <= h_val <= 38) and (40 <= s_val) and (40 <= v_val)
                    is_green = (38 < h_val <= 85) and (30 <= s_val) and (30 <= v_val)
                    
                    if not (is_yellow or is_green):
                        # It's a defect (mold, rot, or gap)
                        # Mold heuristic: light grey / white (low saturation, high value)
                        is_mold = (s_val < 70) and (v_val > 90)
                        # Rot heuristic: dark brown / black (very low value or brown hues)
                        is_rot = (v_val < 70) or ((h_val < 22 or h_val > 155) and s_val > 30 and v_val < 110)
                        
                        if is_mold or is_rot:
                            refined_mask[y, x] = 3  # Diseased (Blue)
                        else:
                            refined_mask[y, x] = 2  # Missing / Gap (Red)
                    else:
                        # Color is healthy yellow/green, ensure it's not background
                        if refined_mask[y, x] == 0:
                            refined_mask[y, x] = 1  # Healthy (Green)
                            
        return refined_mask

    def predict_single(self, img_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Runs inference on a single image.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
                - mask (H, W): integer array with class indices
                - overlay (H, W, 3): blended RGB overlay
                - prob_map (C, H, W): softmax probability map
                - confidence: mean confidence of segmented non-background pixels
        """
        img_path = Path(img_path)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"Could not read image: {img_path}")
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess(img_rgb)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            mask = np.argmax(probs, axis=0).astype(np.uint8)
            
        # Resize original image to config size for overlay alignment
        img_resized = cv2.resize(img_rgb, global_config.input_size)
        
        # Refine mask using CV heuristics to capture mold/rot color exceptions
        mask = self.refine_mask_with_cv(img_resized, mask)
        
        overlay = generate_overlay(img_resized, mask)
        
        # Calculate mean confidence of prediction (non-background pixels)
        corn_pixels_mask = (mask > 0)
        if np.any(corn_pixels_mask):
            pred_probs = np.max(probs, axis=0)
            confidence = float(np.mean(pred_probs[corn_pixels_mask]))
        else:
            confidence = float(np.mean(np.max(probs, axis=0)))
            
        return mask, overlay, probs, confidence

    def predict_folder(self, input_dir: Path, output_dir: Path) -> List[Dict[str, any]]:
        """
        Runs inference on all images in a folder and saves output overlays and JSON results.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        image_paths = list(input_dir.glob("*.jpg"))
        logger.info(f"Running batch prediction on {len(image_paths)} images from {input_dir}")
        
        for img_path in image_paths:
            try:
                mask, overlay, _, conf = self.predict_single(img_path)
                
                # Save visual results
                dest_overlay = output_dir / f"{img_path.stem}_overlay.png"
                cv2.imwrite(str(dest_overlay), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                
                # Extract segmentation percentages
                total_pixels = mask.size
                background_pct = float(np.sum(mask == 0) / total_pixels * 100)
                healthy_pct = float(np.sum(mask == 1) / total_pixels * 100)
                missing_pct = float(np.sum(mask == 2) / total_pixels * 100)
                diseased_pct = float(np.sum(mask == 3) / total_pixels * 100)
                
                res_dict = {
                    "filename": img_path.name,
                    "confidence": conf,
                    "background_percentage": background_pct,
                    "healthy_percentage": healthy_pct,
                    "missing_percentage": missing_pct,
                    "diseased_percentage": diseased_pct
                }
                
                results.append(res_dict)
                
                # Save individual JSON result
                json_path = output_dir / f"{img_path.stem}_result.json"
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(res_dict, jf, indent=4)
                    
            except Exception as e:
                logger.error(f"Error predicting image {img_path.name}: {e}")
                
        # Save aggregated batch results
        agg_json_path = output_dir / "batch_prediction_results.json"
        with open(agg_json_path, "w", encoding="utf-8") as jf:
            json.dump(results, jf, indent=4)
            
        logger.info(f"Batch prediction completed. Saved results to {output_dir}")
        return results
