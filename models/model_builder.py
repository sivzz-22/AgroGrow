"""
AgroGrow — Upgraded Segmentation Model
========================================
Uses segmentation_models_pytorch (SMP) U-Net with a ResNet34 encoder
pre-trained on ImageNet. This replaces the scratch-trained CornNet encoder
with one that already understands textures, edges and shapes, giving a
significant accuracy boost on small datasets.

The model is fully backward-compatible with the existing predictor.py —
it takes (B, 3, 512, 512) tensors and returns (B, 4, 512, 512) logits.

Classes:
  0 — Background
  1 — Healthy kernel
  2 — Missing kernel socket
  3 — Diseased kernel
"""

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    _SMP_AVAILABLE = True
except ImportError:
    _SMP_AVAILABLE = False

from AgroGrow.models.corn_net import CornNet  # fallback


def build_model(num_classes: int = 4,
                encoder: str = "resnet34",
                encoder_weights: str = "imagenet",
                use_smp: bool = True) -> nn.Module:
    """
    Build the best available segmentation model.

    Priority:
      1. SMP U-Net + ResNet34 (ImageNet pretrained) — best accuracy
      2. Legacy CornNet (scratch) — fallback if SMP not installed

    Args:
        num_classes     : number of segmentation classes (4)
        encoder         : SMP encoder name  (resnet34 | resnet50 | efficientnet-b3)
        encoder_weights : pretrained weights ('imagenet' or None)
        use_smp         : set False to force CornNet legacy mode

    Returns:
        nn.Module with forward(x) -> logits of shape (B, num_classes, H, W)
    """
    if use_smp and _SMP_AVAILABLE:
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None,           # raw logits — loss applies softmax internally
            decoder_attention_type="scse",   # Squeeze-and-Excitation attention in decoder
        )
        print(f"[Model] SMP U-Net + {encoder} ({encoder_weights}) — "
              f"{sum(p.numel() for p in model.parameters()):,} params")
        return model
    else:
        if use_smp:
            print("[Model] SMP not available — falling back to CornNet")
        model = CornNet(num_classes=num_classes)
        print(f"[Model] CornNet (scratch) — "
              f"{sum(p.numel() for p in model.parameters()):,} params")
        return model


def load_weights_flexible(model: nn.Module, ckpt_path, device) -> nn.Module:
    """
    Load a checkpoint into model, tolerating mismatched keys (for switching
    between CornNet and SMP architectures on the same checkpoint path).
    """
    import torch
    from pathlib import Path
    path = Path(ckpt_path)
    if not path.exists():
        print(f"[Model] No checkpoint at {path}. Using random init.")
        return model
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    result = model.load_state_dict(state, strict=False)
    missing   = [k for k in result.missing_keys   if "encoder" not in k]
    unexpected = result.unexpected_keys
    if missing:
        print(f"[Model] Missing keys (non-encoder): {len(missing)}")
    if unexpected:
        print(f"[Model] Unexpected keys: {len(unexpected)}")
    print(f"[Model] Loaded weights from {path.name}")
    return model
