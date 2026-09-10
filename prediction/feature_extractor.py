"""
AgroGrow Feature Extraction Module.
Extracts agronomic quality features from segmentation masks:
Corn Area, Defect Area, Percentages, Bounding Box, Kernel Density, and Quality Score.
"""

import cv2
import numpy as np
from typing import Dict, Tuple
from AgroGrow.config import global_config
from AgroGrow.utils.logger import logger

class CornFeatureExtractor:
    """
    Analyzes segmentation masks to calculate corn quality statistics.
    """
    @staticmethod
    def extract_features(mask: np.ndarray) -> Dict[str, any]:
        """
        Extracts features from the segmentation mask.
        
        Args:
            mask (np.ndarray): The segmentation mask of shape (H, W) where:
                               0=Background, 1=Healthy, 2=Missing, 3=Diseased.
                               
        Returns:
            Dict[str, any]: Map containing calculated quality features.
        """
        # Pixel counts
        healthy_pixels = np.sum(mask == 1)
        missing_pixels = np.sum(mask == 2)
        diseased_pixels = np.sum(mask == 3)
        
        total_corn_pixels = healthy_pixels + missing_pixels + diseased_pixels
        defect_pixels = missing_pixels + diseased_pixels
        
        # Calculate percentages relative to total corn area
        if total_corn_pixels > 0:
            healthy_pct = float(healthy_pixels / total_corn_pixels * 100)
            missing_pct = float(missing_pixels / total_corn_pixels * 100)
            disease_pct = float(diseased_pixels / total_corn_pixels * 100)
        else:
            healthy_pct = 0.0
            missing_pct = 0.0
            disease_pct = 0.0
            
        # Calculate Bounding Box of the corn cob
        corn_coords = np.argwhere(mask > 0)
        if len(corn_coords) > 0:
            # corn_coords is in [y, x] format
            y_min, x_min = np.min(corn_coords, axis=0)
            y_max, x_max = np.max(corn_coords, axis=0)
            bbox = (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))
        else:
            bbox = (0, 0, 0, 0)
            
        # Kernel Density estimation
        # Count distinct healthy kernel blobs using connected components
        healthy_mask = (mask == 1).astype(np.uint8)
        num_labels, labels_im = cv2.connectedComponents(healthy_mask)
        # Subtract background component
        num_healthy_blobs = max(0, num_labels - 1)
        
        if bbox[2] * bbox[3] > 0:
            kernel_density = float(num_healthy_blobs / (bbox[2] * bbox[3]) * 1000)  # blobs per 1000 pixels
        else:
            kernel_density = 0.0
            
        # Compute dynamic Quality Score out of 100
        # Standard deduction formula: penalize diseased kernels heavily, missing moderately
        raw_score = healthy_pct - (1.8 * disease_pct) - (1.0 * missing_pct)
        quality_score = float(np.clip(raw_score, 0.0, 100.0))
        
        features = {
            "healthy_percentage": healthy_pct,
            "missing_percentage": missing_pct,
            "disease_percentage": disease_pct,
            "total_corn_area_pixels": int(total_corn_pixels),
            "defect_area_pixels": int(defect_pixels),
            "bounding_box": bbox,  # (x, y, w, h)
            "num_healthy_blobs": num_healthy_blobs,
            "kernel_density_score": kernel_density,
            "quality_score": quality_score,
            "no_corn_detected": (total_corn_pixels == 0)
        }
        
        logger.debug(f"Extracted features: Healthy: {healthy_pct:.2f}%, "
                     f"Disease: {disease_pct:.2f}%, Quality Score: {quality_score:.2f}")
        return features
