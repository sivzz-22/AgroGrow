"""
AgroGrow — Offline Dataset Augmentation
========================================
Expands the training set from ~971 images → ~9,700 images (7× multiplier)
by applying 7 distinct augmentation pipelines to every training image.

Val and Test sets are NOT touched — they must remain clean for evaluation.

Usage:
    python -m AgroGrow.utils.augment_dataset
    python -m AgroGrow.utils.augment_dataset --multiplier 5
    python -m AgroGrow.utils.augment_dataset --dry-run      # count only, no write

Output: augmented images saved alongside originals in dataset/images/train/
        augmented masks  saved alongside originals in dataset/masks/train/
        Augmented files are prefixed with  aug{N}_  so originals are untouched.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

from AgroGrow.config import global_config
from AgroGrow.utils.logger import logger


# ── 7 augmentation pipelines — each produces one new sample per original ────

def _build_pipelines(H: int, W: int):
    """Return a list of named (name, Compose) augmentation pipelines."""
    shared_resize = A.Resize(H, W)

    pipelines = [
        # 1. Horizontal flip + slight brightness
        ("hflip_bright", A.Compose([
            shared_resize,
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.15, p=0.8),
            A.GaussNoise(std_range=(0.01, 0.03), p=0.5),
        ])),

        # 2. Rotation + scale
        ("rotate_scale", A.Compose([
            shared_resize,
            A.ShiftScaleRotate(
                shift_limit=0.08, scale_limit=0.15, rotate_limit=20,
                border_mode=cv2.BORDER_REFLECT_101, p=1.0
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, p=0.6),
        ])),

        # 3. Colour jitter (simulates different lighting / camera sensors)
        ("color_jitter", A.Compose([
            shared_resize,
            A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.RandomGamma(gamma_limit=(80, 130), p=0.7),
            A.CLAHE(clip_limit=3.0, p=0.5),
        ])),

        # 4. Elastic distortion (simulates curved/rolled corn ear)
        ("elastic", A.Compose([
            shared_resize,
            A.ElasticTransform(alpha=60, sigma=8, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.10, p=0.5),
        ])),

        # 5. Blur + noise (simulates low-quality phone camera / motion blur)
        ("blur_noise", A.Compose([
            shared_resize,
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=7),
                A.MedianBlur(blur_limit=5),
            ], p=0.9),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.12, p=0.5),
        ])),

        # 6. Random shadow + crop-zoom (simulates outdoor field photos)
        ("shadow_zoom", A.Compose([
            shared_resize,
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2), p=0.7),
            A.RandomResizedCrop(size=(H, W), scale=(0.75, 1.0), ratio=(0.8, 1.2), p=0.8),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=15, p=0.6),
        ])),

        # 7. Vertical flip + coarse dropout (simulates rotated/partial cob images)
        ("vflip_dropout", A.Compose([
            shared_resize,
            A.VerticalFlip(p=1.0),
            A.CoarseDropout(
                num_holes_range=(2, 6), hole_height_range=(20, 60), hole_width_range=(20, 60),
                fill=0, p=0.6
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, p=0.5),
        ])),
    ]
    return pipelines


def augment_split(split: str, multiplier: int, dry_run: bool) -> int:
    """Augment one dataset split. Returns count of new files written."""
    images_dir = global_config.images_dir / split
    masks_dir  = global_config.masks_dir  / split

    # Only process originals (not previously augmented files)
    orig_images = sorted([
        p for p in images_dir.glob("*.jpg")
        if not p.stem.startswith("aug")
    ])

    if not orig_images:
        logger.warning(f"No original images found in {images_dir}")
        return 0

    H, W = global_config.input_size
    pipelines = _build_pipelines(H, W)[:multiplier]  # cap at requested multiplier

    logger.info(f"[{split}] {len(orig_images)} originals x {len(pipelines)} pipelines "
                f"= {len(orig_images) * len(pipelines)} new samples")

    if dry_run:
        return len(orig_images) * len(pipelines)

    written = 0
    for img_path in tqdm(orig_images, desc=f"Augmenting [{split}]"):
        mask_path = masks_dir / f"{img_path.stem}.png"

        img  = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        for aug_name, pipeline in pipelines:
            try:
                result    = pipeline(image=img, mask=mask)
                aug_img   = result["image"]
                aug_mask  = result["mask"]

                # Ensure mask values remain in [0, num_classes-1]
                aug_mask  = np.clip(aug_mask, 0, global_config.num_classes - 1).astype(np.uint8)

                out_name  = f"aug{aug_name}_{img_path.stem}"
                out_img_p = images_dir / f"{out_name}.jpg"
                out_msk_p = masks_dir  / f"{out_name}.png"

                cv2.imwrite(str(out_img_p), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                cv2.imwrite(str(out_msk_p), aug_mask)
                written += 1

            except Exception as e:
                logger.warning(f"Augmentation '{aug_name}' failed on {img_path.name}: {e}")

    logger.info(f"[{split}] Written {written} augmented samples.")
    return written


def clean_augmented(split: str):
    """Remove all previously augmented files (prefix aug*) from a split."""
    images_dir = global_config.images_dir / split
    masks_dir  = global_config.masks_dir  / split
    removed = 0
    for d in [images_dir, masks_dir]:
        for p in d.glob("aug*"):
            p.unlink()
            removed += 1
    logger.info(f"[{split}] Removed {removed} augmented files.")


def main():
    ap = argparse.ArgumentParser(description="AgroGrow Offline Dataset Augmentation")
    ap.add_argument("--multiplier", type=int, default=7,
                    help="Number of augmented variants per original image (max 7, default 7)")
    ap.add_argument("--split", type=str, default="train",
                    help="Which split to augment: train | val | test (default: train)")
    ap.add_argument("--clean", action="store_true",
                    help="Remove previously generated aug* files first")
    ap.add_argument("--dry-run", action="store_true",
                    help="Count how many samples would be generated without writing")
    args = ap.parse_args()

    multiplier = max(1, min(7, args.multiplier))
    split = args.split

    logger.info("=" * 60)
    logger.info("  AgroGrow — Offline Dataset Augmentation")
    logger.info("=" * 60)
    logger.info(f"  Split      : {split}")
    logger.info(f"  Multiplier : {multiplier}x (each original -> {multiplier} new samples)")
    logger.info(f"  Dry run    : {args.dry_run}")

    if args.clean and not args.dry_run:
        logger.info(f"Cleaning existing aug* files from [{split}]...")
        clean_augmented(split)

    total = augment_split(split, multiplier, args.dry_run)

    if args.dry_run:
        orig = len([p for p in (global_config.images_dir / split).glob("*.jpg")
                    if not p.stem.startswith("aug")])
        logger.info(f"\nDry-run result: {orig} originals x {multiplier} = {total} new samples")
        logger.info(f"Total training set after augmentation: {orig + total} images")
    else:
        logger.info(f"\nDone. Total new augmented samples: {total}")
        orig = len([p for p in (global_config.images_dir / split).glob("*.jpg")
                    if not p.stem.startswith("aug")])
        aug  = len([p for p in (global_config.images_dir / split).glob("aug*.jpg")])
        logger.info(f"Training set: {orig} originals + {aug} augmented = {orig + aug} total")


if __name__ == "__main__":
    main()
