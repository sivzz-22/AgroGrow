"""
AgroGrow — Research Paper Model Training
=========================================
Trains a DEDICATED CornNet for single-variety commercial corn segmentation,
exactly matching the research paper benchmark setup.

Differences from the general model (train.py):
  * Saves to  weights/paper_model.pth  (not best_model.pth)
  * Uses paper-exact augmentation (no VerticalFlip, conservative colour jitter)
  * Higher early-stopping patience (20) for deeper convergence
  * Stronger class weights for disease and missing classes
  * No variety-classifier integration — pure model only

Usage:
    python train_paper.py
    python train_paper.py --from-scratch
    python train_paper.py --patience 25
    python train_paper.py --lr 5e-5
"""

import os, sys, json, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch, cv2, numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from AgroGrow.config import global_config
from AgroGrow.utils.logger import logger
from AgroGrow.models.corn_net import CornNet
from AgroGrow.models.loss import HybridLoss
from AgroGrow.utils.metrics import MetricsEvaluator


# ---------------------------------------------------------------------------
# Paper-exact Dataset
# ---------------------------------------------------------------------------
class PaperCornDataset(Dataset):
    """Conservative augmentation matching the original paper protocol."""
    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)

    def __init__(self, split: str, augment: bool = False):
        self.images_dir  = global_config.images_dir / split
        self.masks_dir   = global_config.masks_dir  / split
        self.image_paths = sorted(self.images_dir.glob("*.jpg"))
        if not self.image_paths:
            logger.warning(f"No images found for split '{split}'")

        H, W = global_config.input_size
        base = [A.Resize(H, W), A.Normalize(self.MEAN, self.STD), ToTensorV2()]

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.08,
                                   rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.4),
                A.GaussNoise(p=0.3),
                *base,
            ])
        else:
            self.transform = A.Compose(base)

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        ip = self.image_paths[idx]
        mp = self.masks_dir / f"{ip.stem}.png"
        img  = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = np.clip(mask, 0, global_config.num_classes - 1).astype(np.uint8)
        out  = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].long(), ip.name


# ---------------------------------------------------------------------------
# Paper Trainer
# ---------------------------------------------------------------------------
class PaperTrainer:
    PAPER_MODEL_PATH      = global_config.weights_dir / "paper_model.pth"
    PAPER_CHECKPOINT_PATH = global_config.weights_dir / "paper_checkpoint.pth"
    HISTORY_PATH          = global_config.results_dir / "paper_training_history.json"

    def __init__(self, model, train_loader, val_loader,
                 criterion, optimizer, lr_scheduler, device):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.lr_scheduler = lr_scheduler
        self.device       = device
        self.scaler       = GradScaler("cuda", enabled=(device.type == "cuda"))
        self.evaluator    = MetricsEvaluator(num_classes=global_config.num_classes)
        self.best_miou    = 0.0
        self.start_epoch  = 0
        self.history      = {"train_loss": [], "val_loss": [],
                             "train_miou": [], "val_miou": []}

    def load_checkpoint(self):
        if self.PAPER_CHECKPOINT_PATH.exists():
            ckpt = torch.load(self.PAPER_CHECKPOINT_PATH,
                              map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.best_miou   = ckpt.get("best_miou", 0.0)
            if "history" in ckpt:
                self.history = ckpt["history"]
            logger.info(f"Resumed paper checkpoint epoch={self.start_epoch-1}, "
                        f"best_miou={self.best_miou:.4f}")
            return True
        return False

    def init_from_best(self):
        best_path = global_config.weights_dir / "best_model.pth"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Pre-initialized paper model weights from {best_path.name} for paper-focused training.")
            return True
        logger.warning(f"Weights file not found at {best_path}. Starting with random initialization.")
        return False

    def _run_epoch(self, loader, train: bool):
        self.model.train() if train else self.model.eval()
        total_loss, cm = 0.0, np.zeros(
            (self.evaluator.num_classes, self.evaluator.num_classes), dtype=np.int64)
        ctx  = torch.enable_grad() if train else torch.no_grad()
        desc = "Train" if train else "Val  "
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
        metrics = self.evaluator.evaluate_from_confusion_matrix(cm)
        mean_loss = total_loss / len(loader.dataset)
        return mean_loss, float(metrics.get("mean_iou", 0.0))

    def fit(self, num_epochs=None, patience: int = 20):
        no_improve, epoch = 0, self.start_epoch
        logger.info(f"Paper training: epoch={epoch}, patience={patience}, "
                    f"max={num_epochs if num_epochs else 'unlimited'}")
        while True:
            if num_epochs is not None and epoch >= num_epochs:
                logger.info(f"Reached max epochs ({num_epochs}).")
                break
            logger.info(f"\nEpoch {epoch + 1}")
            train_loss, train_miou = self._run_epoch(self.train_loader, True)
            val_loss,   val_miou   = self._run_epoch(self.val_loader,   False)
            self.lr_scheduler.step(val_loss)
            for k, v in [("train_loss", train_loss), ("val_loss", val_loss),
                         ("train_miou", train_miou), ("val_miou", val_miou)]:
                self.history[k].append(v)
            logger.info(f"  Train  loss={train_loss:.4f}  mIoU={train_miou:.4f}")
            logger.info(f"  Val    loss={val_loss:.4f}    mIoU={val_miou:.4f}")
            torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_miou": self.best_miou, "history": self.history},
                       self.PAPER_CHECKPOINT_PATH)
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                            "val_miou": val_miou}, self.PAPER_MODEL_PATH)
                logger.info(f"  >> New best paper model  val_mIoU={val_miou:.4f}")
                no_improve = 0
            else:
                no_improve += 1
                logger.info(f"  No improvement {no_improve}/{patience} "
                            f"(auto-stop in {patience - no_improve} epochs)")
                if no_improve >= patience:
                    logger.info(f"  Early stop. Best mIoU={self.best_miou:.4f}")
                    break
            with open(self.HISTORY_PATH, "w") as f:
                json.dump(self.history, f, indent=2)
            epoch += 1
        logger.info(f"\nTraining complete. Best Val mIoU: {self.best_miou:.4f}")
        logger.info(f"Saved -> {self.PAPER_MODEL_PATH}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Train Research Paper CornNet")
    p.add_argument("--from-scratch", action="store_true", help="Start training from pure random weights (paper default)")
    p.add_argument("--init-from-best", action="store_true", help="Warm-start training from weights/best_model.pth for faster convergence")
    p.add_argument("--epochs",      type=int,   default=None, help="Max epochs (default: None - auto-stops by patience)")
    p.add_argument("--patience",    type=int,   default=20,   help="Early stopping patience (default: 20)")
    p.add_argument("--batch-size",  type=int,   default=2,    help="Batch size (default: 2)")
    p.add_argument("--lr",          type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    args = p.parse_args()

    logger.info("=" * 60)
    logger.info("  AgroGrow — Research Paper Model Training")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    train_ds = PaperCornDataset("train", augment=True)
    val_ds   = PaperCornDataset("val",   augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    logger.info(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    model = CornNet(num_classes=global_config.num_classes)
    logger.info(f"CornNet params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Paper-exact loss: heavier weight on disease + missing detection
    criterion = HybridLoss(
        class_weights=[0.05, 1.0, 10.0, 5.0],  # [bg, healthy, missing, diseased]
        wce_weight=0.5,
        dice_weight=0.5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    trainer = PaperTrainer(model, train_loader, val_loader,
                           criterion, optimizer, scheduler, device)

    if args.from_scratch:
        logger.info("Training from scratch (clean random weights).")
    else:
        resumed = trainer.load_checkpoint()
        if not resumed:
            if args.init_from_best:
                trainer.init_from_best()
            else:
                logger.info("No prior paper checkpoint found. Starting fresh from scratch.")

    trainer.fit(num_epochs=args.epochs, patience=args.patience)


if __name__ == "__main__":
    main()
