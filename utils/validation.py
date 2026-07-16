"""
AgroGrow Dataset Validation Module.
Performs verification checks on the prepared dataset splits
and generates a validation report.
"""

import hashlib
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from AgroGrow.config import global_config
from AgroGrow.utils.logger import logger

class DatasetValidator:
    """
    Executes dataset checks and compiles the final validation report.
    """
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.images_dir = dataset_dir / "images"
        self.masks_dir = dataset_dir / "masks"
        self.report_path = global_config.reports_dir / "dataset_validation_report.md"

    def compute_md5(self, file_path: Path) -> str:
        """Computes MD5 hash of a file."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def validate(self) -> Tuple[bool, str]:
        """
        Runs validation checks on the images and masks splits.
        
        Returns:
            Tuple[bool, str]: (is_valid_flag, message_or_report_path)
        """
        logger.info("Executing dataset validation checks...")
        
        errors: List[str] = []
        warnings: List[str] = []
        report_lines: List[str] = [
            "# AgroGrow Dataset Validation Report",
            f"**Validation Date/Time**: 2026-07-14",
            "",
            "## Summary of Validation Results",
            ""
        ]
        
        splits = ["train", "val", "test"]
        stats: Dict[str, Dict[str, int]] = {
            s: {"images": 0, "masks": 0, "valid_pairs": 0, "size_mismatch": 0, "invalid_labels": 0} for s in splits
        }
        
        seen_hashes = {}
        duplicates_count = 0
        corrupted_count = 0
        
        for split in splits:
            split_img_dir = self.images_dir / split
            split_mask_dir = self.masks_dir / split
            
            if not split_img_dir.exists() or not split_mask_dir.exists():
                errors.append(f"Directory for split '{split}' is missing.")
                continue
                
            img_files = sorted(list(split_img_dir.glob("*.jpg")))
            stats[split]["images"] = len(img_files)
            
            for img_path in img_files:
                # 1. Verification of matching mask
                mask_path = split_mask_dir / f"{img_path.stem}.png"
                if not mask_path.exists():
                    errors.append(f"[{split}] Missing mask for image: {img_path.name}")
                    continue
                stats[split]["masks"] += 1
                
                # 2. Check for duplicate image files
                img_hash = self.compute_md5(img_path)
                if img_hash in seen_hashes:
                    duplicates_count += 1
                    warnings.append(f"[{split}] Duplicate file detected: {img_path.name} is a duplicate of {seen_hashes[img_hash]}")
                else:
                    seen_hashes[img_hash] = f"{split}/{img_path.name}"
                
                # 3. Read image and mask to check corruption & dimensions
                img = cv2.imread(str(img_path))
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    corrupted_count += 1
                    errors.append(f"[{split}] Image is corrupted or unreadable: {img_path.name}")
                    continue
                if mask is None:
                    corrupted_count += 1
                    errors.append(f"[{split}] Mask is corrupted or unreadable: {mask_path.name}")
                    continue
                
                # 4. Check shape matching
                img_h, img_w, _ = img.shape
                mask_h, mask_w = mask.shape
                
                if (img_h, img_w) != global_config.input_size:
                    warnings.append(f"[{split}] Image {img_path.name} size {img.shape[:2]} does not match config {global_config.input_size}.")
                if (img_h, img_w) != (mask_h, mask_w):
                    stats[split]["size_mismatch"] += 1
                    errors.append(f"[{split}] Dimension mismatch: image {img_path.name} {img.shape[:2]} vs mask {mask_path.name} {mask.shape}")
                    continue
                
                # 5. Check labels matching
                unique_labels = np.unique(mask)
                invalid_labels = [l for l in unique_labels if l >= global_config.num_classes]
                if invalid_labels:
                    stats[split]["invalid_labels"] += 1
                    errors.append(f"[{split}] Mask {mask_path.name} contains invalid labels: {invalid_labels}")
                    continue
                    
                stats[split]["valid_pairs"] += 1
                
        # Build Report
        report_lines.append("| Split | Total Images | Total Masks | Valid Pairs | Dimension Mismatches | Invalid Mask Labels |")
        report_lines.append("| --- | --- | --- | --- | --- | --- |")
        for split in splits:
            s_data = stats[split]
            report_lines.append(f"| {split.capitalize()} | {s_data['images']} | {s_data['masks']} | {s_data['valid_pairs']} | {s_data['size_mismatch']} | {s_data['invalid_labels']} |")
            
        report_lines.append("\n## Quality Check Summary")
        report_lines.append(f"- **Total Unique Images**: {len(seen_hashes)}")
        report_lines.append(f"- **Total Duplicate Images Skipped**: {duplicates_count}")
        report_lines.append(f"- **Total Corrupted Files Detected**: {corrupted_count}")
        
        # Check overall validity
        is_valid = len(errors) == 0
        
        if is_valid:
            report_lines.append("\n### STATUS: PASSED")
            report_lines.append("No critical validation errors were detected in the dataset. Ready for model training.")
        else:
            report_lines.append("\n### STATUS: FAILED")
            report_lines.append(f"Validation failed with {len(errors)} critical errors. See list below:")
            report_lines.append("\n### Critical Errors")
            for err in errors:
                report_lines.append(f"- [ERROR] {err}")
                
        if warnings:
            report_lines.append("\n### Warnings")
            for warn in warnings:
                report_lines.append(f"- [WARNING] {warn}")
                
        # Write report to file
        with open(self.report_path, "w", encoding="utf-8") as rf:
            rf.write("\n".join(report_lines))
            
        logger.info(f"Validation report saved to {self.report_path}")
        
        if not is_valid:
            logger.critical("Dataset validation failed! Halting pipeline execution.")
            return False, f"Validation Failed: Found {len(errors)} critical errors. Report written to {self.report_path}"
            
        return True, str(self.report_path)
