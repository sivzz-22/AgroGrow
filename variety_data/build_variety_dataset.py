"""
AgroGrow Variety Dataset Builder.
Downloads corn variety images from free public sources using icrawler (Bing/Google image search)
and organises them into labelled train/val splits for MobileNet variety classifier training.

Varieties collected:
  0 - dent_corn     (Yellow/White Dent)
  1 - flint_corn    (Indian / Multicoloured Flint)
  2 - sweet_corn    (Fresh Sweet Corn)
  3 - popcorn       (Popcorn kernels / cobs)
  4 - blue_corn     (Blue / Black Hopi corn)

Usage:
    python variety_data/build_variety_dataset.py --images_per_class 150 --output_dir variety_data/images
"""

import os
import sys
import shutil
import argparse
import random
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

VARIETY_QUERIES = {
    "dent_corn": [
        "yellow dent corn cob close up",
        "yellow corn kernels macro",
        "white dent corn cob agricultural",
        "commercial corn cob yellow grain",
        "corn on the cob yellow kernels"
    ],
    "flint_corn": [
        "Indian corn multicoloured cob",
        "flint corn ruby red purple kernels",
        "Glass Gem corn cob colorful",
        "Bloody Butcher corn dark red",
        "ornamental corn cob multicolor",
        "Painted Mountain corn",
        "calico corn decorative"
    ],
    "sweet_corn": [
        "sweet corn cob pale yellow",
        "fresh sweet corn kernels",
        "baby sweet corn close up",
        "cream corn cob macro photo",
        "bicolor sweet corn yellow white"
    ],
    "popcorn": [
        "popcorn kernels unpopped macro",
        "popcorn corn cob hard kernels",
        "strawberry popcorn red cob",
        "pearl white popcorn cob",
        "miniature popcorn cob close up"
    ],
    "blue_corn": [
        "blue corn cob hopi",
        "black corn dark kernels",
        "blue corn kernels close up",
        "Hopi blue corn cob",
        "dark purple corn cob",
        "black aztec corn",
        "blue tortilla corn kernels"
    ]
}

VARIETY_LABELS = list(VARIETY_QUERIES.keys())


def download_variety_images(output_dir: Path, images_per_class: int = 150):
    """Downloads images for each variety using Bing image crawler."""
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        print("ERROR: icrawler not installed. Run: pip install icrawler")
        return False

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  AgroGrow Variety Dataset Builder")
    print(f"  Target: {images_per_class} images per class")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    for variety, queries in VARIETY_QUERIES.items():
        class_dir = output_dir / "raw" / variety
        class_dir.mkdir(parents=True, exist_ok=True)
        existing = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        needed = images_per_class - len(existing)

        if needed <= 0:
            print(f"[SKIP] {variety}: already has {len(existing)} images.")
            continue

        print(f"\n[DOWNLOADING] {variety} --- need {needed} more images...")
        per_query = max(1, needed // len(queries) + 5)

        for query in queries:
            crawler = BingImageCrawler(
                storage={"root_dir": str(class_dir)},
                feeder_threads=2,
                parser_threads=2,
                downloader_threads=4
            )
            try:
                crawler.crawl(
                    keyword=query,
                    max_num=per_query,
                    min_size=(150, 150),
                    file_idx_offset="auto"
                )
            except Exception as e:
                print(f"  Warning: query '{query}' failed: {e}")
            
            # Check if we have enough
            found = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            if len(found) >= images_per_class:
                break

        final_count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
        print(f"  [OK] {variety}: {final_count} images downloaded")

    return True


def split_into_train_val(raw_dir: Path, split_dir: Path, val_ratio: float = 0.2):
    """Splits raw images into train/ and val/ subdirectories."""
    raw_dir  = Path(raw_dir)
    split_dir = Path(split_dir)

    for split in ["train", "val"]:
        (split_dir / split).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Splitting into train/val ({int((1-val_ratio)*100)}%/{int(val_ratio*100)}%)")
    print(f"{'='*60}")

    total_train, total_val = 0, 0

    for variety in VARIETY_LABELS:
        src = raw_dir / variety
        if not src.exists():
            print(f"  [SKIP] {variety}: no raw images found.")
            continue

        imgs = list(src.glob("*.jpg")) + list(src.glob("*.png")) + list(src.glob("*.jpeg"))
        random.shuffle(imgs)
        n_val   = max(1, int(len(imgs) * val_ratio))
        n_train = len(imgs) - n_val

        (split_dir / "train" / variety).mkdir(parents=True, exist_ok=True)
        (split_dir / "val"   / variety).mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(imgs):
            dest_split = "val" if i < n_val else "train"
            shutil.copy2(img, split_dir / dest_split / variety / img.name)

        print(f"  {variety}: {n_train} train, {n_val} val")
        total_train += n_train
        total_val   += n_val

    print(f"\n  Total: {total_train} train  |  {total_val} val")
    print(f"  Output directory: {split_dir}")


def main():
    parser = argparse.ArgumentParser(description="AgroGrow Variety Dataset Builder")
    parser.add_argument("--images_per_class", type=int, default=150,
                        help="Target images per variety class (default: 150)")
    parser.add_argument("--output_dir", type=str,
                        default="d:/Sem_7/research paper 1/AgroGrow/variety_data",
                        help="Output directory for downloaded images")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download step (only split existing raw images)")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Validation set fraction (default: 0.2)")
    args = parser.parse_args()

    base   = Path(args.output_dir)
    raw    = base / "raw"
    splits = base / "splits"

    if not args.skip_download:
        success = download_variety_images(raw, args.images_per_class)
        if not success:
            return

    split_into_train_val(raw, splits, args.val_ratio)

    print(f"\n{'='*60}")
    print(f"  Dataset ready at: {splits}")
    print(f"  Next step: python variety_data/train_variety_classifier.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
