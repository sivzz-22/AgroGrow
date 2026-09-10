"""
AgroGrow Corn Variety Classifier — Inference Module.
Uses a fine-tuned MobileNetV3-Small to classify corn variety from an image crop.
Falls back to colour-grading if the trained weights are not yet available.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple
from torchvision import transforms, models
import torch.nn as nn

from AgroGrow.config import global_config
from AgroGrow.utils.logger import logger

VARIETY_DISPLAY_FALLBACK = {
    "dent_corn":  "Commercial Dent Corn (Zea mays indentata)",
    "flint_corn": "Indian / Multi-Colored Flint Corn (Zea mays indurata)",
    "sweet_corn": "Sweet Corn (Zea mays saccharata)",
    "popcorn":    "Popcorn (Zea mays everta)",
    "blue_corn":  "Blue / Black Corn (Hopi Blue — Zea mays)"
}

INFER_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class VarietyClassifier:
    """
    MobileNetV3-Small corn variety classifier.
    If weights are available, uses deep learning.
    Otherwise signals that the fallback colour system should be used.
    """

    WEIGHTS_PATH = global_config.weights_dir / "variety_classifier.pth"
    CONFIDENCE_THRESHOLD = 0.50  # below this → report uncertain / fall back

    def __init__(self, device: Optional[str] = None):
        self._model = None
        self._idx_to_class = None
        self._variety_display = None
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._load()

    def _load(self):
        if not self.WEIGHTS_PATH.exists():
            logger.info("Variety classifier weights not found — colour-grading fallback active.")
            return
        try:
            ckpt = torch.load(str(self.WEIGHTS_PATH), map_location=self._device, weights_only=False)
            num_classes = len(ckpt["idx_to_class"])

            model = models.mobilenet_v3_small(weights=None)
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, num_classes)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self._device).eval()

            self._model           = model
            self._idx_to_class    = ckpt["idx_to_class"]   # {0: "dent_corn", ...}
            self._variety_display = ckpt.get("variety_display", VARIETY_DISPLAY_FALLBACK)
            logger.info(f"Variety classifier loaded: {num_classes} classes, "
                        f"val_acc={ckpt.get('val_acc', '?'):.1f}%")
        except Exception as e:
            logger.warning(f"Failed to load variety classifier: {e}. Using colour-grading fallback.")
            self._model = None

    @property
    def is_trained(self) -> bool:
        """True if the MobileNet model weights are loaded."""
        return self._model is not None

    def classify(self, img_rgb: np.ndarray) -> Tuple[str, str, float]:
        """
        Classifies corn variety from an RGB image crop.

        Args:
            img_rgb: H×W×3 numpy array in RGB order.

        Returns:
            (class_key, display_name, confidence)
            Returns ('unknown', 'Unknown', 0.0) if model not loaded.
        """
        if self._model is None:
            return "unknown", "Unknown", 0.0

        tensor = INFER_TRANSFORMS(img_rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        idx         = int(np.argmax(probs))
        confidence  = float(probs[idx])
        class_key   = self._idx_to_class.get(idx, "dent_corn")
        display_name = self._variety_display.get(class_key,
                            VARIETY_DISPLAY_FALLBACK.get(class_key, class_key))

        if confidence < self.CONFIDENCE_THRESHOLD:
            logger.debug(f"Variety confidence too low ({confidence:.2f}) → falling back to colour-grading.")
            return "uncertain", display_name, confidence

        logger.info(f"Variety classified: {display_name} (confidence: {confidence:.2%})")
        return class_key, display_name, confidence


# ── Singleton accessor ────────────────────────────────────────────────────────
_classifier_instance: Optional[VarietyClassifier] = None

def get_variety_classifier() -> VarietyClassifier:
    """Returns a cached singleton VarietyClassifier."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = VarietyClassifier()
    return _classifier_instance
