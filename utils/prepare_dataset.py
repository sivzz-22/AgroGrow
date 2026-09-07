"""
AgroGrow Dataset Preparation Module — Model 4.
Generates ground-truth masks with correct class assignments:
  Class 0 — Background  (soil, leaves, sky, hands)
  Class 1 — Healthy     (intact yellow/orange/white corn kernels)
  Class 2 — Missing     (dark empty sockets INSIDE the cob hull)
  Class 3 — Diseased    (rotten/mouldy kernels — dark brown or fuzzy grey)
"""

import os
import json
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
    Handles file discovery, duplicate detection, corruption checking,
    split organisation, and pseudo-mask generation.
    """
    def __init__(self, raw_dir: Path, target_dir: Path):
        self.raw_dir = raw_dir
        self.target_dir = target_dir
        self.images_dest = target_dir / "images"
        self.masks_dest  = target_dir / "masks"
        for split in ["train", "val", "test"]:
            (self.images_dest / split).mkdir(parents=True, exist_ok=True)
            (self.masks_dest  / split).mkdir(parents=True, exist_ok=True)

    # ─── helpers ─────────────────────────────────────────────────────────────

    def compute_md5(self, file_path: Path) -> str:
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def parse_labelme_json(self, json_path: Path,
                           target_size: Tuple[int, int]) -> Tuple[np.ndarray, bool]:
        """Rasterises Labelme polygon annotations into a class mask."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            orig_h = data.get("imageHeight", target_size[0])
            orig_w = data.get("imageWidth",  target_size[1])
            mask   = np.zeros((orig_h, orig_w), dtype=np.uint8)
            sx, sy = target_size[1] / float(orig_w), target_size[0] / float(orig_h)
            lmap   = {
                # ── Healthy / Good corn ───────────────────────────────────────
                "healthy": 1, "corn": 1, "healthy_corn": 1, "good": 1,
                "goodcorn": 1, "good_corn": 1, "rawcorn": 1, "field_corn": 1,
                "normal": 1, "intact": 1, "kernel": 1,
                # ── Missing kernel sockets ────────────────────────────────────
                "missing": 2, "missing_grain": 2, "missing_kernel": 2,
                "empty": 2, "socket": 2, "gap": 2,
                # ── Diseased / Damaged ────────────────────────────────────────
                "disease": 3, "diseased": 3, "mold": 3, "rot": 3,
                "damaged": 3, "defective": 3, "bad": 3, "badcorn": 3,
                "rotten": 3, "mould": 3, "infected": 3,
                # ── Background ────────────────────────────────────────────────
                "background": 0, "bg": 0, "other": 0
            }
            for shape in data.get("shapes", []):
                label = shape.get("label", "").lower().strip()
                cls   = lmap.get(label, 1)
                pts   = np.array(shape.get("points", []), dtype=np.float32)
                if len(pts) > 0:
                    pts[:, 0] *= sx; pts[:, 1] *= sy
                    cv2.fillPoly(mask, [np.int32(pts)], cls)
            return cv2.resize(mask, (target_size[1], target_size[0]),
                              interpolation=cv2.INTER_NEAREST), True
        except Exception as e:
            logger.warning(f"Labelme JSON parse failed ({json_path}): {e}")
            return np.zeros(target_size, dtype=np.uint8), False

    # ─── mask generation ─────────────────────────────────────────────────────

    def generate_pseudo_mask(self, img_path: Path) -> Tuple[np.ndarray, bool]:
        """
        Priority:
          1. Use Labelme JSON if present (manual annotations).
          2. Otherwise generate a CV-based pseudo-mask at 512×512.

        Class definitions (FIXED for Model 4):
          0 — Background : everything outside the corn ear hull
          1 — Healthy    : yellow / orange / white intact kernels
          2 — Missing    : dark empty sockets inside the hull
                           detected by black-hat ONLY within the yellow region
          3 — Diseased   : dark-value rotten kernels OR bright-grey mould
        """
        # ── 1. Try Labelme JSON first ──────────────────────────────────────
        json_path = img_path.with_suffix(".json")
        if json_path.exists():
            mask, ok = self.parse_labelme_json(json_path, global_config.input_size)
            if ok:
                return mask, True

        # ── 2. CV pseudo-mask ─────────────────────────────────────────────
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return np.zeros(global_config.input_size, dtype=np.uint8), False

        H, W = global_config.input_size          # (512, 512)
        img  = cv2.resize(img_bgr, (W, H))
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]
        L_ch = lab[:, :, 0]

        # ── Step 1 : segment ONLY corn kernel colours ─────────────────────
        # Yellow / orange kernels  (Hue 8–38, decent saturation & brightness)
        mask_yellow = cv2.inRange(hsv,
                                  np.array([8,  40, 50]),
                                  np.array([38, 255, 255]))

        # Reddish / magenta diseased kernels
        mask_red = cv2.inRange(hsv,
                               np.array([0, 60, 40]),
                               np.array([10, 255, 255]))

        # White / cream sweet-corn — tight thresholds to avoid sky
        mask_white = ((L_ch > 160) & (s_ch < 35) & (v_ch > 150)).astype(np.uint8) * 255

        # Combined kernel colours
        kernel_color = cv2.bitwise_or(mask_yellow,
                       cv2.bitwise_or(mask_red, mask_white))

        # ── Step 2 : build cob hull ───────────────────────────────────────
        # Close small gaps within the cob (shadow lines between kernel rows)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        kernel_closed = cv2.morphologyEx(kernel_color, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(kernel_closed,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cob_hull = np.zeros((H, W), dtype=np.uint8)
        if contours:
            sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)
            max_area    = cv2.contourArea(sorted_cnts[0])
            for cnt in sorted_cnts:
                area = cv2.contourArea(cnt)
                # Keep only blobs that are ≥15 % of the largest blob's area
                # and at least 1200 px — avoids stray dots
                if area >= 1200 and area >= 0.15 * max_area:
                    hull_pts = cv2.convexHull(cnt)
                    cv2.drawContours(cob_hull, [hull_pts], -1, 1, -1)

        if np.sum(cob_hull) == 0:
            return np.zeros((H, W), dtype=np.uint8), True

        cob_idx = cob_hull > 0

        # ── Step 3 : build label map ──────────────────────────────────────
        label_map = np.zeros((H, W), dtype=np.uint8)
        label_map[cob_idx] = 1                   # default inside hull → Healthy

        # ── Step 4 : Diseased (Class 3) ──────────────────────────────────
        # 3a. Dark-value rotten / burnt kernels inside hull
        is_rot  = (v_ch < 60) & cob_idx
        # 3b. Fuzzy grey/white mould — high lightness BUT very low saturation
        #     Use tighter thresholds than before to avoid catching sky blobs
        is_mold = (L_ch > 175) & (s_ch < 30) & (v_ch > 130) & cob_idx
        disease = is_rot | is_mold
        label_map[disease] = 3

        # ── Step 5 : Missing (Class 2) ────────────────────────────────────
        # KEY FIX from Model 3:
        #   Missing sockets are dark gaps INSIDE the yellow kernel region.
        #   We run black-hat ONLY on the yellow sub-mask (not ~yellow).
        #   Black-hat highlights dark valleys between bright kernel rows.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use a kernel slightly larger than one kernel cell (~15 px at 512 res)
        k_bh  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bhat  = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_bh)

        # Threshold: dark valleys brighter than 20 (avoids solid black disease)
        _, bh_thresh = cv2.threshold(bhat, 20, 255, cv2.THRESH_BINARY)

        # Missing: inside hull, NOT already diseased, AND inside the yellow region
        # (inter-kernel gaps sit between yellow rows → they are NOT yellow themselves)
        yellow_bool = mask_yellow.astype(bool)
        # A missing socket pixel is: dark valley inside hull, not disease, not a
        # yellow kernel, not a white kernel
        white_bool  = mask_white.astype(bool)
        missing = ((bh_thresh > 0) & cob_idx & ~disease
                   & ~yellow_bool & ~white_bool)
        label_map[missing] = 2

        return label_map, True

    # ─── main processing loop ─────────────────────────────────────────────────

    def process(self) -> Dict[str, List[str]]:
        logger.info("Starting dataset preprocessing and pseudo-mask generation...")
        stats  = {"copied": [], "corrupted": [], "duplicates": []}
        hashes = set()

        for split in ["train", "val", "test"]:
            src = self.raw_dir / "data" / split
            if not src.exists():
                logger.warning(f"Source folder not found: {src}")
                continue

            for fp in src.rglob("*.jpg"):
                h = self.compute_md5(fp)
                if h in hashes:
                    stats["duplicates"].append(str(fp)); continue
                hashes.add(h)

                mask, ok = self.generate_pseudo_mask(fp)
                if not ok:
                    stats["corrupted"].append(str(fp))
                    logger.error(f"Corrupted: {fp}"); continue

                img     = cv2.imread(str(fp))
                W, H    = global_config.input_size[1], global_config.input_size[0]
                img_rs  = cv2.resize(img, (W, H))

                dst_img  = self.images_dest / split / fp.name
                dst_mask = self.masks_dest  / split / f"{fp.stem}.png"
                cv2.imwrite(str(dst_img),  img_rs)
                cv2.imwrite(str(dst_mask), mask)
                stats["copied"].append(str(dst_img))

        logger.info(
            f"Dataset preparation completed. "
            f"Copied: {len(stats['copied'])}, "
            f"Corrupted: {len(stats['corrupted'])}, "
            f"Duplicates: {len(stats['duplicates'])}"
        )
        return stats
