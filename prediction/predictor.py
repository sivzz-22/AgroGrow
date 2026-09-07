"""
AgroGrow Segmentation Prediction Module — Model 4.
Auto-crops to the corn ear bounding box before inference so that outdoor
background (sky, leaves, soil) is stripped before the network sees it.
"""

import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from AgroGrow.config import global_config
from AgroGrow.models.corn_net import CornNet
from AgroGrow.utils.logger import logger
from AgroGrow.utils.dataset import generate_overlay


def auto_crop_corn_ear(img_rgb: np.ndarray,
                       pad_frac: float = 0.05) -> Tuple[np.ndarray, Tuple]:
    """
    Detects the corn ear bounding box using kernel texture energy and multi-variety
    color cues, returning a tight crop around the corn ear, stripping outdoor field
    foliage, husks, and soil.
    """
    H_img, W_img = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    R = img_rgb[:, :, 0].astype(float)
    G = img_rgb[:, :, 1].astype(float)
    B = img_rgb[:, :, 2].astype(float)

    # 1. Kernel Texture Energy (Sobel cross-gradient energy)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    energy = cv2.blur(gx * gy, (21, 21))
    norm_e = energy / (np.max(energy) + 1e-6)

    # 2. Multi-Variety Corn Color Cues (Yellow, White, Ruby, Bronze, Purple)
    m_yellow = (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 50) & (R > B * 1.35)
    m_white  = (lab[:, :, 0] > 145) & (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 135)
    m_ruby   = ((hsv[:, :, 0] <= 15) | (hsv[:, :, 0] >= 165)) & (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 30)
    m_bronze = (hsv[:, :, 0] <= 30) & (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 28) & (hsv[:, :, 2] <= 135) & (R > B * 1.2)
    m_purple = (hsv[:, :, 0] >= 120) & (hsv[:, :, 0] <= 165) & (hsv[:, :, 1] >= 40)
    corn_cues = (m_yellow | m_white | m_ruby | m_bronze | m_purple)

    # 3. Outdoor Foliage / Husk / Mud Rejection
    rej_green = ((hsv[:, :, 0] >= 30) & (hsv[:, :, 0] <= 90) & (G >= R * 0.90)) | ((hsv[:, :, 0] >= 36) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] >= 30))
    rej_husk  = (R - B < 65) & (hsv[:, :, 1] < 95) & (hsv[:, :, 2] > 75) & (G > B * 1.05)
    rej_mud   = (hsv[:, :, 2] < 28) | ((R < 60) & (G < 60) & (B < 60) & (norm_e < 0.15))
    rej_sky   = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] >= 25)

    cob_core = corn_cues & (~rej_green) & (~rej_husk) & (~rej_mud) & (~rej_sky) & (norm_e > 0.08)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cob_closed = cv2.morphologyEx(cob_core.astype(np.uint8), cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(cob_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_rgb, (0, 0, W_img, H_img)

    sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    max_area    = cv2.contourArea(sorted_cnts[0])

    all_pts = []
    for cnt in sorted_cnts:
        area = cv2.contourArea(cnt)
        if area >= 0.10 * max_area and area >= 500:
            all_pts.append(cnt)

    if not all_pts:
        all_pts = [sorted_cnts[0]]

    merged = np.vstack(all_pts)
    x, y, w, h = cv2.boundingRect(merged)

    # Add small padding
    pad_x = int(w * pad_frac)
    pad_y = int(h * pad_frac)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W_img, x + w + pad_x)
    y2 = min(H_img, y + h + pad_y)

    cropped = img_rgb[y1:y2, x1:x2]
    if cropped.size == 0:
        return img_rgb, (0, 0, W_img, H_img)

    return cropped, (x1, y1, x2 - x1, y2 - y1)


def clean_prediction_mask(img_rgb: np.ndarray,
                          raw_mask: np.ndarray,
                          probs: Optional[np.ndarray] = None,
                          fg_conf_threshold: float = 0.35,
                          corn_variety: str = "auto") -> Tuple[np.ndarray, str]:
    """
    Precision post-processing:
      1. Spatial Cob Region-of-Interest (ROI) Extraction:
         Combines 2D kernel lattice texture energy with multi-variety color cues.
         Strictly sets all background pixels outside the cob to Class 0 (Background / Black).
      2. Rejection of sunlit field leaves, dried straw husks, shadows, and mud.
      3. Corn Variety Identification & Healthy Pigmented Kernel Mapping:
         Detects Indian / Multi-colored Flint Corn (Zea mays indurata).
         In Flint varieties, naturally ruby-red, burgundy, bronze, and purple kernels
         are classified as Class 1 (Healthy Grain, Green), NOT diseased/defective!
      4. Morphological speck removal.
    """
    h, w = raw_mask.shape
    if img_rgb.shape[:2] != (h, w):
        img_rgb = cv2.resize(img_rgb, (w, h))

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    R = img_rgb[:, :, 0].astype(float)
    G = img_rgb[:, :, 1].astype(float)
    B = img_rgb[:, :, 2].astype(float)

    # 1. Texture Energy
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    energy = cv2.blur(gx * gy, (21, 21))
    norm_e = energy / (np.max(energy) + 1e-6)

    # 2. Multi-Variety Corn Color Cues
    m_yellow = (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 50) & (R > B * 1.35)
    m_white  = (lab[:, :, 0] > 145) & (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 135)
    m_ruby   = ((hsv[:, :, 0] <= 15) | (hsv[:, :, 0] >= 165)) & (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 30)
    m_bronze = (hsv[:, :, 0] <= 30) & (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 28) & (hsv[:, :, 2] <= 135) & (R > B * 1.2)
    m_purple = (hsv[:, :, 0] >= 120) & (hsv[:, :, 0] <= 165) & (hsv[:, :, 1] >= 40)
    corn_cues = (m_yellow | m_white | m_ruby | m_bronze | m_purple)

    # 3. Outdoor Foliage / Dried Husk / Mud / Sky Rejection
    rej_green = ((hsv[:, :, 0] >= 30) & (hsv[:, :, 0] <= 90) & (G >= R * 0.90)) | ((hsv[:, :, 0] >= 36) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] >= 30))
    rej_husk  = (R - B < 65) & (hsv[:, :, 1] < 95) & (hsv[:, :, 2] > 75) & (G > B * 1.05)
    rej_mud   = (hsv[:, :, 2] < 28) | ((R < 60) & (G < 60) & (B < 60) & (norm_e < 0.15))
    rej_sky   = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] >= 25)

    # 4. Cob Region-of-Interest (ROI) Isolation
    cob_core = corn_cues & (~rej_green) & (~rej_husk) & (~rej_mud) & (~rej_sky) & (norm_e > 0.08)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cob_closed = cv2.morphologyEx(cob_core.astype(np.uint8), cv2.MORPH_CLOSE, k)

    cob_roi = np.zeros((h, w), dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cob_closed, connectivity=8)
    if num_labels > 1:
        max_area = np.max(stats[1:, cv2.CC_STAT_AREA])
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= 0.08 * max_area and stats[lbl, cv2.CC_STAT_AREA] >= 250:
                cob_roi[labels == lbl] = 1
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        cob_roi = cv2.dilate(cob_roi, k_dilate)
    else:
        cob_roi = np.ones((h, w), dtype=np.uint8)

    # Restrict raw mask to cob ROI and suppress background noise
    clean_mask = raw_mask.copy()
    clean_mask[cob_roi == 0] = 0
    clean_mask[rej_green | rej_husk | rej_mud | rej_sky] = 0

    # Softmax confidence filtering
    if probs is not None and probs.shape[0] >= 4:
        bg_prob = probs[0]
        fg_max_prob = np.max(probs[1:4], axis=0)
        weak_fg = (fg_max_prob < fg_conf_threshold) | (bg_prob >= fg_max_prob * 0.95)
        clean_mask[weak_fg] = 0

    # 5. Grain Variety Determination & Mapping
    is_pigmented = (m_ruby | m_bronze | m_purple)
    pigmented_px = np.sum(is_pigmented & (cob_roi == 1))
    total_cob_px = np.sum(corn_cues & (cob_roi == 1))
    p_ratio = pigmented_px / max(1, total_cob_px)

    v_lower = corn_variety.lower()
    is_flint_active = ("flint" in v_lower) or ("indian" in v_lower) or ("pigment" in v_lower) or \
                      ("auto" in v_lower and p_ratio >= 0.15)

    detected_variety = "Indian / Multi-Colored Flint Corn (Zea mays indurata)" if is_flint_active else "Commercial Dent Corn (Yellow/White)"

    if is_flint_active:
        # In Indian / Flint corn, naturally ruby-red, bronze, and purple kernels are HEALTHY grain, NOT defect!
        flint_kernel_mask = is_pigmented & (cob_roi == 1) & (norm_e > 0.08)
        clean_mask[flint_kernel_mask] = 1

    # 6. Morphological cleanup of stray noise specks
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_m = (clean_mask > 0).astype(np.uint8)
    fg_clean = cv2.morphologyEx(fg_m, cv2.MORPH_OPEN, k_small)
    clean_mask[fg_clean == 0] = 0

    return clean_mask, detected_variety


class CornPredictor:
    """Loads CornNet weights and performs inference with automatic CPU fallback."""

    def __init__(self, weights_path: Optional[Path] = None, device: Optional[str] = None):
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")
            if torch.cuda.is_available():
                try:
                    # Probe CUDA to ensure driver context is alive (prevents sleep/wake crashes)
                    _ = torch.zeros(1, device="cuda")
                    self.device = torch.device("cuda")
                except Exception as e:
                    logger.warning(f"CUDA context probe failed ({e}). Falling back to CPU.")
                    self.device = torch.device("cpu")

        self.model = CornNet(num_classes=global_config.num_classes)
        self.last_detected_variety = "Standard Dent Corn (Yellow/White)"

        path = weights_path or (global_config.weights_dir / "best_model.pth")
        if path.exists():
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded predictor model weights from: {path} (device: {self.device})")
        else:
            logger.warning(f"Weights not found at {path}. Using random init.")

        self.model.to(self.device).eval()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, img_rgb: np.ndarray) -> torch.Tensor:
        """Resize → normalise → tensor."""
        W, H = global_config.input_size[1], global_config.input_size[0]
        resized  = cv2.resize(img_rgb, (W, H))
        normed   = (resized.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(normed).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def predict_single(self,
                       img_path: Union[str, Path],
                       auto_crop: bool = True,
                       corn_variety: str = "auto",
                       black_background: bool = True
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Full pipeline:
          1. Load image
          2. Auto-crop to corn ear if requested (strips outdoor background)
          3. Run CornNet inference (with transparent CPU fallback if CUDA has issues)
          4. Post-processing to suppress background bleed and handle corn varieties
          5. Generate colour overlay (Green=Healthy, Blue=Missing, Red=Diseased, Black=Background)
          6. Return mask, overlay, prob_map, confidence
        """
        img_path = Path(img_path)
        img_bgr  = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ── Auto-crop to corn ear ────────────────────────────────────────
        if auto_crop:
            img_cropped, bbox = auto_crop_corn_ear(img_rgb)
        else:
            img_cropped = img_rgb

        # ── Inference on cropped image (with automatic CPU fallback) ─────
        tensor = self._preprocess(img_cropped)
        try:
            with torch.no_grad():
                logits = self.model(tensor)
                probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                raw_mask = np.argmax(probs, axis=0).astype(np.uint8)
        except RuntimeError as e:
            if "cuda" in str(e).lower():
                logger.warning(f"CUDA runtime error encountered: {e}. Falling back to CPU for inference.")
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
                tensor = tensor.to(self.device)
                with torch.no_grad():
                    logits = self.model(tensor)
                    probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    raw_mask = np.argmax(probs, axis=0).astype(np.uint8)
            else:
                raise e

        # ── Post-processing to remove background bleed and handle variety ─
        W, H = global_config.input_size[1], global_config.input_size[0]
        img_display = cv2.resize(img_cropped, (W, H))
        mask, variety_name = clean_prediction_mask(img_display, raw_mask, probs, corn_variety=corn_variety)
        self.last_detected_variety = variety_name

        # ── Build overlay with high-contrast black background (Zero Bleed) 
        overlay     = generate_overlay(img_display, mask, black_background=black_background)

        # ── Confidence ───────────────────────────────────────────────────
        corn_px = (mask > 0)
        if np.any(corn_px):
            confidence = float(np.mean(np.max(probs, axis=0)[corn_px]))
        else:
            confidence = float(np.mean(np.max(probs, axis=0)))

        return mask, overlay, probs, confidence

    def predict_folder(self,
                       input_dir: Path,
                       output_dir: Path) -> List[Dict]:
        """Batch inference on a folder of images."""
        input_dir  = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        imgs    = list(input_dir.glob("*.jpg"))
        logger.info(f"Batch prediction on {len(imgs)} images.")

        for img_path in imgs:
            try:
                mask, overlay, _, conf = self.predict_single(img_path)
                cv2.imwrite(
                    str(output_dir / f"{img_path.stem}_overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                )
                tot = mask.size
                res = {
                    "filename":            img_path.name,
                    "confidence":          conf,
                    "background_pct":      float(np.sum(mask == 0) / tot * 100),
                    "healthy_pct":         float(np.sum(mask == 1) / tot * 100),
                    "missing_pct":         float(np.sum(mask == 2) / tot * 100),
                    "diseased_pct":        float(np.sum(mask == 3) / tot * 100),
                }
                results.append(res)
                with open(output_dir / f"{img_path.stem}_result.json", "w") as jf:
                    json.dump(res, jf, indent=4)
            except Exception as e:
                logger.error(f"Error on {img_path.name}: {e}")

        agg_path = output_dir / "batch_results.json"
        with open(agg_path, "w") as jf:
            json.dump(results, jf, indent=4)
        logger.info(f"Batch done. Results in {output_dir}")
        return results
