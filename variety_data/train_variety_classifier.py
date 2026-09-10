"""
AgroGrow MobileNetV3 Corn Variety Classifier — Trainer.
Fine-tunes a pre-trained MobileNetV3-Small on 5 corn variety classes.

Classes:
  0 - dent_corn    (Yellow/White Dent — most common commercial)
  1 - flint_corn   (Indian / Multicoloured Flint — ruby, purple, bronze)
  2 - sweet_corn   (Fresh pale sweet corn)
  3 - popcorn      (Small hard kernels — popcorn)
  4 - blue_corn    (Blue / Black Hopi corn)

Usage:
    python variety_data/train_variety_classifier.py --data_dir variety_data/splits --epochs 25
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from AgroGrow.utils.logger import logger

VARIETY_CLASSES = ["dent_corn", "flint_corn", "sweet_corn", "popcorn", "blue_corn"]
VARIETY_DISPLAY = {
    "dent_corn":  "Commercial Dent Corn (Zea mays indentata)",
    "flint_corn": "Indian / Multi-Colored Flint Corn (Zea mays indurata)",
    "sweet_corn": "Sweet Corn (Zea mays saccharata)",
    "popcorn":    "Popcorn (Zea mays everta)",
    "blue_corn":  "Blue / Black Corn (Hopi Blue — Zea mays)"
}

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def build_model(num_classes: int = 5, freeze_backbone: bool = True) -> nn.Module:
    """Returns a fine-tuned MobileNetV3-Small."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    
    if freeze_backbone:
        # Freeze all layers except the classifier
        for param in model.features.parameters():
            param.requires_grad = False
    
    # Replace the final classifier
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV3 corn variety classifier")
    parser.add_argument("--data_dir",  type=str,
                        default="d:/Sem_7/research paper 1/AgroGrow/variety_data/splits")
    parser.add_argument("--output_dir", type=str,
                        default="d:/Sem_7/research paper 1/AgroGrow/weights")
    parser.add_argument("--epochs",    type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--unfreeze_epoch", type=int, default=10,
                        help="After this epoch, unfreeze backbone for fine-tuning")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"

    if not train_dir.exists():
        print(f"ERROR: Training directory not found: {train_dir}")
        print("Run build_variety_dataset.py first to download images.")
        return

    # Count available images
    total_train = sum(1 for _ in train_dir.rglob("*.jpg"))
    total_val   = sum(1 for _ in val_dir.rglob("*.jpg"))
    print(f"\n{'='*60}")
    print(f"  AgroGrow MobileNetV3 Variety Classifier Training")
    print(f"  Training images: {total_train} | Validation: {total_val}")
    print(f"  Classes: {VARIETY_CLASSES}")
    print(f"{'='*60}\n")

    # Datasets
    train_dataset = datasets.ImageFolder(str(train_dir), transform=TRAIN_TRANSFORMS)
    val_dataset   = datasets.ImageFolder(str(val_dir),   transform=VAL_TRANSFORMS)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model     = build_model(num_classes=len(VARIETY_CLASSES), freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone at specified epoch for full fine-tuning
        if epoch == args.unfreeze_epoch:
            print(f"\n  [Epoch {epoch}] Unfreezing backbone for full fine-tuning...")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = val_epoch(  model, val_loader,   criterion, device)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc, 2),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_acc, 2)
        }
        history.append(record)

        print(f"  Epoch [{epoch:3d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}%  |  "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%  "
              f"{'  ← BEST' if val_acc > best_val_acc else ''}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": train_dataset.class_to_idx,
                "idx_to_class": {v: k for k, v in train_dataset.class_to_idx.items()},
                "variety_display": VARIETY_DISPLAY,
                "val_acc": val_acc,
                "epoch": epoch
            }, str(output_dir / "variety_classifier.pth"))
            logger.info(f"Saved best variety classifier: val_acc={val_acc:.1f}%")

    # Save history
    with open(output_dir / "variety_classifier_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best Validation Accuracy: {best_val_acc:.1f}%")
    print(f"  Model saved: {output_dir / 'variety_classifier.pth'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
