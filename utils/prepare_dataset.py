"""
AgroGrow Dataset Preparation Module.
Parses raw images, generates HSV-based pseudo-segmentation masks, and builds
the standard directory structure for training Corn-Net.
"""

import os
import shutil
import hashlib
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from AgroGrow.config import global_config
from AgroGrow.utils.logger import logger

class DatasetPreparer:
    """
    Handles file discovery, duplicate detection, corruption checking, split organization,
    and pseudo-mask generation using computer vision heuristics.
    """
    def __init__(self, raw_dir: Path, target_dir: Path):
        self.raw_dir = raw_dir
        self.target_dir = target_dir
        self.images_dest = target_dir / "images"
        self.masks_dest = target_dir / "masks"
        
        # Ensure directories exist
        for split in ["train", "val", "test"]:
            (self.images_dest / split).mkdir(parents=True, exist_ok=True)
            (self.masks_dest / split).mkdir(parents=True, exist_ok=True)

    def compute_md5(self, file_path: Path) -> str:
        """Computes MD5 hash of a file to check for duplicates."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def generate_pseudo_mask(self, img_path: Path) -> Tuple[np.ndarray, bool]:
        """
        Loads image, detects corn using HSV color segmentation,
        injects synthetic missing and diseased regions, and outputs mask.
        
        Returns:
            Tuple[np.ndarray, bool]: (mask, success_flag)
        """
        img = cv2.imread(str(img_path))
        if img is None:
            return np.zeros(global_config.input_size, dtype=np.uint8), False
            
        # Resize to standardized input size
        img = cv2.resize(img, global_config.input_size)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Segment yellow/green/orange corn cobs
        # Hue ranges: Yellow (10-35), Green (35-85), Orange/Red (5-15)
        lower_yellow = np.array([10, 40, 40])
        upper_yellow = np.array([40, 255, 255])
        
        lower_green = np.array([40, 30, 30])
        upper_green = np.array([85, 255, 255])
        
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Combined corn mask
        corn_mask = cv2.bitwise_or(mask_yellow, mask_green)
        
        # Find contours of corn_mask
        contours, _ = cv2.findContours(corn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cob_hull = np.zeros_like(corn_mask)
        for cnt in contours:
            if cv2.contourArea(cnt) > 2000:  # Ignore small noise
                hull = cv2.convexHull(cnt)
                cv2.drawContours(cob_hull, [hull], -1, 1, -1)
                
        # Create output labels map: 0 = Background, 1 = Healthy Corn
        label_map = np.zeros(global_config.input_size, dtype=np.uint8)
        
        # Classify pixels inside cob boundaries
        for y in range(global_config.input_size[0]):
            for x in range(global_config.input_size[1]):
                if cob_hull[y, x] > 0:
                    h_val, s_val, v_val = hsv[y, x]
                    
                    is_yellow = (10 <= h_val <= 38) and (40 <= s_val) and (40 <= v_val)
                    is_green = (38 < h_val <= 85) and (30 <= s_val) and (30 <= v_val)
                    
                    if not (is_yellow or is_green):
                        # Defect (mold or rot)
                        is_mold = (s_val < 70) and (v_val > 90)
                        is_rot = (v_val < 70) or ((h_val < 22 or h_val > 155) and s_val > 30 and v_val < 110)
                        if is_mold or is_rot:
                            label_map[y, x] = 3  # Diseased
                        else:
                            label_map[y, x] = 2  # Missing
                    else:
                        label_map[y, x] = 1  # Healthy
        
        # Find coordinates of healthy corn pixels to place extra synthetic defects
        corn_coords = np.argwhere(label_map == 1)
        if len(corn_coords) > 100:
            # Deterministic pseudo-random generation based on image name hash
            seed_val = int(hashlib.md5(img_path.name.encode()).hexdigest(), 16) % 10000
            rng = np.random.default_rng(seed=seed_val)
            
            # Inject 2 to 5 missing kernel spots (Class 2)
            num_missing = rng.integers(2, 6)
            for _ in range(num_missing):
                healthy_coords = np.argwhere(label_map == 1)
                if len(healthy_coords) > 10:
                    idx = rng.choice(len(healthy_coords))
                    cy, cx = healthy_coords[idx]
                    radius = rng.integers(6, 12)
                    cv2.circle(label_map, (cx, cy), radius, 2, -1)
                
            # Inject 1 to 3 diseased kernel spots (Class 3)
            num_diseased = rng.integers(1, 4)
            for _ in range(num_diseased):
                healthy_coords = np.argwhere(label_map == 1)
                if len(healthy_coords) > 10:
                    idx = rng.choice(len(healthy_coords))
                    cy, cx = healthy_coords[idx]
                    radius = rng.integers(5, 10)
                    cv2.circle(label_map, (cx, cy), radius, 3, -1)
                    
        return label_map, True

    def process(self) -> Dict[str, List[str]]:
        """
        Iterates over the raw dataset splits, copies files, and creates masks.
        
        Returns:
            Dict[str, List[str]]: Statistics of processing results.
        """
        logger.info("Starting dataset preprocessing and pseudo-mask generation...")
        stats = {"copied": [], "corrupted": [], "duplicates": []}
        hashes = set()
        
        splits = ["train", "val", "test"]
        for split in splits:
            split_src_dir = self.raw_dir / "data" / split
            if not split_src_dir.exists():
                logger.warning(f"Source split folder {split_src_dir} does not exist.")
                continue
                
            # Search for JPG files in subfolders (corn, fruit_and_vegetable corn)
            for file_path in split_src_dir.rglob("*.jpg"):
                # 1. Check for duplicates
                file_hash = self.compute_md5(file_path)
                if file_hash in hashes:
                    stats["duplicates"].append(str(file_path))
                    logger.debug(f"Duplicate file skipped: {file_path.name}")
                    continue
                hashes.add(file_hash)
                
                # 2. Check for corruption & generate pseudo-mask
                mask, success = self.generate_pseudo_mask(file_path)
                if not success:
                    stats["corrupted"].append(str(file_path))
                    logger.error(f"Corrupted or invalid image: {file_path}")
                    continue
                
                # 3. Save copy of image (standardized resize)
                img = cv2.imread(str(file_path))
                img_resized = cv2.resize(img, global_config.input_size)
                
                dest_img_path = self.images_dest / split / file_path.name
                dest_mask_path = self.masks_dest / split / f"{file_path.stem}.png"
                
                cv2.imwrite(str(dest_img_path), img_resized)
                cv2.imwrite(str(dest_mask_path), mask)
                
                stats["copied"].append(str(dest_img_path))
                
        logger.info(f"Dataset preparation completed. Copied: {len(stats['copied'])}, "
                    f"Corrupted: {len(stats['corrupted'])}, Duplicates: {len(stats['duplicates'])}")
        return stats
