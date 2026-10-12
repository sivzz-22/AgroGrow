"""
AgroGrow — Upgraded Model Training Script
==========================================
Trains a U-Net with ResNet34 (ImageNet pretrained) encoder on the
augmented dataset. Saves to weights/paper_model.pth (single-variety mode).

Key improvements over train_paper.py:
  • ResNet34 ImageNet encoder  →  massive feature quality boost
  • Stronger augmentations     →  rotation, elastic, colour jitter, shadows
  • CosineAnnealingWarmRestarts LR schedule  →  better convergence
  • Gradient clipping           →  training stability
  • Per-class IoU logging        →  see exactly which class is weak

Usage:
    python train_upgraded.py                    # train upgraded model
    python train_upgraded.py --encoder resnet50 # use larger encoder
    python train_upgraded.py --from-scratch     # no imagenet init
    python train_upgraded.py --epochs 60        # fixed epoch count
    python train_upgraded.py --resume           # resume from checkpoint
"""

import os, sys, json, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from AgroGrow.config import global_config
from AgroGrow.utils.logger import logger
from AgroGrow.models.model_builder import build_model
from AgroGrow.models.loss import HybridLoss
from AgroGrow.utils.metrics import MetricsEvaluator


# ── Dataset ─────────────────────────────────────────────────────────────────

class CornDataset(Dataset):
    """Dataset that loads from dataset/images/{split}/ and dataset/masks/{split}/"""
    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)

    def __init__(self, split: str, augment: bool = False):
        self.images_dir  = global_config.images_dir / split
        self.masks_dir   = global_config.masks_dir  / split
        self.image_paths = sorted(self.images_dir.glob("*.jpg"))

        if not self.image_paths:
            logger.warning(f"No images found for split '{split}' in {self.images_dir}")

        H, W = global_config.input_size
        base = [A.Resize(H, W), A.Normalize(self.MEAN, self.STD), ToTensorV2()]

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.06, scale_limit=0.12, rotate_limit=15,
                    border_mode=cv2.BORDER_REFLECT_101, p=0.5
                ),
                A.OneOf([
                    A.ElasticTransform(alpha=50, sigma=7, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
                ], p=0.35),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25,
                                     val_shift_limit=20, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.20,
                                           contrast_limit=0.15, p=0.5),
                A.RandomGamma(gamma_limit=(80, 130), p=0.3),
                A.RandomShadow(p=0.25),
                A.CLAHE(clip_limit=3.0, p=0.25),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MotionBlur(blur_limit=5),
                ], p=0.25),
                A.GaussNoise(std_range=(0.005, 0.025), p=0.3),
                *base,
            ])
        else:
            self.transform = A.Compose(base)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        ip   = self.image_paths[idx]
        mp   = self.masks_dir / f"{ip.stem}.png"
        img  = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = np.clip(mask, 0, global_config.num_classes - 1).astype(np.uint8)
        out  = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].long(), ip.name


# ── Trainer ─────────────────────────────────────────────────────────────────

class UpgradedTrainer:
    CKPT_PATH  = global_config.weights_dir / "upgraded_checkpoint.pth"
    BEST_PATH  = global_config.weights_dir / "paper_model.pth"   # replaces paper model
    HIST_PATH  = global_config.results_dir / "upgraded_training_history.json"
    CLASS_NAMES = global_config.class_names  # [background, healthy, missing, diseased]

    def __init__(self, model, train_loader, val_loader,
                 criterion, optimizer, scheduler, device):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.device       = device
        self.scaler       = GradScaler("cuda", enabled=(device.type == "cuda"))
        self.evaluator    = MetricsEvaluator(num_classes=global_config.num_classes)
        self.best_miou    = 0.0
        self.start_epoch  = 0
        self.history      = {"train_loss": [], "val_loss": [],
                             "train_miou": [], "val_miou": [],
                             "per_class_iou": []}

    def _save_checkpoint(self, epoch: int):
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_miou":            self.best_miou,
            "history":              self.history,
        }, self.CKPT_PATH)

    def resume(self) -> bool:
        if not self.CKPT_PATH.exists():
            return False
        ckpt = torch.load(self.CKPT_PATH, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_miou   = ckpt.get("best_miou", 0.0)
        if "history" in ckpt:
            self.history = ckpt["history"]
        logger.info(f"Resumed from epoch {self.start_epoch - 1}, best_mIoU={self.best_miou:.4f}")
        return True

    def _run_epoch(self, loader, train: bool):
        self.model.train() if train else self.model.eval()
        total_loss = 0.0
        cm = np.zeros((self.evaluator.num_classes, self.evaluator.num_classes),
                      dtype=np.int64)
        desc = "Train" if train else "Val  "

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for images, masks, _ in tqdm(loader, desc=f"  {desc}", leave=False):
                images = images.to(self.device, non_blocking=True)
                masks  = masks.to(self.device,  non_blocking=True)

                if train:
                    self.optimizer.zero_grad()
                    with autocast("cuda", enabled=(self.device.type == "cuda")):
                        out  = self.model(images)
                        loss = self.criterion(out, masks)
                    self.scaler.scale(loss).backward()
                    # Gradient clipping — prevents exploding gradients
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    with autocast("cuda", enabled=(self.device.type == "cuda")):
                        out  = self.model(images)
                        loss = self.criterion(out, masks)

                total_loss += loss.item() * images.size(0)
                preds = torch.argmax(out, dim=1).detach().cpu().numpy()
                lbls  = masks.detach().cpu().numpy()
                cm   += self.evaluator.compute_confusion_matrix(preds, lbls)

        metrics   = self.evaluator.evaluate_from_confusion_matrix(cm)
        mean_loss = total_loss / len(loader.dataset)
        mean_iou  = float(metrics.get("mean_iou", 0.0))
        per_class = metrics.get("per_class_iou", {})
        return mean_loss, mean_iou, per_class

    def fit(self, num_epochs=None, patience: int = 25):
        no_improve = 0
        epoch      = self.start_epoch

        logger.info(f"Training starts at epoch {epoch}, patience={patience}")

        while True:
            if num_epochs is not None and epoch >= num_epochs:
                logger.info(f"Reached max epochs ({num_epochs}).")
                break

            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}")

            train_loss, train_miou, _  = self._run_epoch(self.train_loader, train=True)
            val_loss,   val_miou,   pc = self._run_epoch(self.val_loader,   train=False)

            self.scheduler.step()   # CosineAnnealingWarmRestarts — step every epoch

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_miou"].append(train_miou)
            self.history["val_miou"].append(val_miou)
            self.history["per_class_iou"].append(pc)

            logger.info(f"  Train  loss={train_loss:.4f}  mIoU={train_miou:.4f}")
            logger.info(f"  Val    loss={val_loss:.4f}    mIoU={val_miou:.4f}")

            # Per-class IoU log — see which class is weak
            for cls_name, iou_val in pc.items():
                logger.info(f"    {cls_name:<12}: IoU = {iou_val:.4f}")

            self._save_checkpoint(epoch)

            if val_miou > self.best_miou:
                self.best_miou = val_miou
                ckpt_payload = {
                    "epoch":             epoch,
                    "model_state_dict":  self.model.state_dict(),
                    "val_miou":          val_miou,
                    "architecture":      "smp_unet_resnet34",
                }
                torch.save(ckpt_payload, self.BEST_PATH)
                torch.save(ckpt_payload, global_config.weights_dir / "best_model.pth")
                logger.info(f"  >> NEW BEST  val_mIoU={val_miou:.4f}  -> saved to {self.BEST_PATH.name} & best_model.pth")
                no_improve = 0
            else:
                no_improve += 1
                logger.info(f"  No improvement {no_improve}/{patience}")
                if no_improve >= patience:
                    logger.info(f"  Early stop. Best val_mIoU={self.best_miou:.4f}")
                    break

            with open(self.HIST_PATH, "w") as f:
                json.dump(self.history, f, indent=2)

            epoch += 1

        logger.info(f"\nTraining complete. Best Val mIoU: {self.best_miou:.4f}")
        logger.info(f"Best model saved → {self.BEST_PATH}")


# ── Entry Point ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="AgroGrow Upgraded Training")
    ap.add_argument("--encoder",      type=str,   default="resnet34",
                    help="SMP encoder: resnet34 | resnet50 | efficientnet-b3 (default: resnet34)")
    ap.add_argument("--no-pretrain",  action="store_true",
                    help="Train without ImageNet weights (not recommended)")
    ap.add_argument("--from-scratch", action="store_true",
                    help="Delete existing checkpoint and start fresh")
    ap.add_argument("--resume",       action="store_true",
                    help="Resume from upgraded_checkpoint.pth if it exists")
    ap.add_argument("--epochs",       type=int,   default=None,
                    help="Max epochs (default: unlimited, stops by patience)")
    ap.add_argument("--patience",     type=int,   default=25,
                    help="Early-stopping patience (default: 25)")
    ap.add_argument("--batch-size",   type=int,   default=4,
                    help="Batch size (default: 4 — increase if GPU allows)")
    ap.add_argument("--lr",           type=float, default=3e-4,
                    help="Initial learning rate (default: 3e-4)")
    ap.add_argument("--workers",      type=int,   default=2,
                    help="DataLoader workers (default: 2)")
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info("  AgroGrow — Upgraded Model Training (ResNet34 + SMP)")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    # ── Data ────────────────────────────────────────────────────────────────
    train_ds = CornDataset("train", augment=True)
    val_ds   = CornDataset("val",   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0)
    )
    logger.info(f"Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

    # ── Model ────────────────────────────────────────────────────────────────
    encoder_weights = None if args.no_pretrain else "imagenet"
    model = build_model(
        num_classes=global_config.num_classes,
        encoder=args.encoder,
        encoder_weights=encoder_weights,
    )

    # ── Loss — heavier weight on rare classes (missing ~2%, diseased ~5%) ───
    criterion = HybridLoss(
        class_weights=[0.05, 1.0, 10.0, 5.0],   # [bg, healthy, missing, diseased]
        wce_weight=0.5,
        dice_weight=0.5,
    )

    # ── Optimiser — separate LR for encoder (pretrained) vs decoder (new) ──
    encoder_params = [p for n, p in model.named_parameters() if "encoder" in n]
    decoder_params = [p for n, p in model.named_parameters() if "encoder" not in n]

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr * 0.1},   # 10× lower for pretrained
        {"params": decoder_params, "lr": args.lr},
    ], weight_decay=1e-4)

    # CosineAnnealingWarmRestarts: soft restarts every T_0 epochs
    T_0 = max(10, (args.epochs or 80) // 4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=2, eta_min=1e-6
    )

    # ── Trainer ─────────────────────────────────────────────────────────────
    trainer = UpgradedTrainer(model, train_loader, val_loader,
                              criterion, optimizer, scheduler, device)

    if args.from_scratch:
        logger.info("--from-scratch: deleting existing checkpoint.")
        ckpt = trainer.CKPT_PATH
        if ckpt.exists():
            ckpt.unlink()
    elif args.resume:
        resumed = trainer.resume()
        if not resumed:
            logger.info("No checkpoint found — starting fresh.")

    trainer.fit(num_epochs=args.epochs, patience=args.patience)


if __name__ == "__main__":
    main()
