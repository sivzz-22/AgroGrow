 # AgroGrow Deep Learning Training Guide

This guide details how to prepare datasets, run deep learning semantic training, and execute storage prediction model training.

## 1. Segmentation Pipeline (PyTorch)

### Step A: Setup Preprocessing and Validation
Because the raw dataset lacks segmentation masks, execute the preprocessor. This copies images, creates matching splits, and generates HSV-based label maps with simulated defect overlays (healthy, missing, diseased classes). It also generates a validation report:
```bash
python AgroGrow/prepare_dataset.py
```
*Verification*: Check `AgroGrow/reports/dataset_validation_report.md` to confirm the splits compiled successfully.

### Step B: Launch Training
Execute the training pipeline for Corn-Net:
```bash
python AgroGrow/train.py
```
*Options and Defaults (Configurable in `AgroGrow/config.py`):*
- **Batch Size**: 8
- **Learning Rate**: 1e-4
- **Epochs**: 15 (with Early Stopping threshold of 5 epochs)
- **Model Output**: Saves the best parameters dynamically to `AgroGrow/weights/best_model.pth`.

### Step C: Check training curves
Open `AgroGrow/results/training_metrics_curves.png` to examine training/validation loss, Mean IoU (mIoU), and Mean Dice score across training cycles.

---

## 2. Storage Life Prediction Pipeline (Scikit-Learn & XGBoost)

### Step A: Synthesize Storage Environmental Database
Run the simulator to construct a database containing environmental variables and storage targets:
```bash
python AgroGrow/storage_prediction/dataset_generator.py
```
*Generates*: `AgroGrow/dataset/storage_data.csv` containing 1,200 simulated warehouse logs.

### Step B: Train Regressors & Save Best
Run training to fit Random Forest and XGBoost regressors, compare their metrics, and serialize the best one:
```bash
python -c "from AgroGrow.storage_prediction.model import train_storage_models; train_storage_models()"
```
*Outputs*:
- Comparison metrics chart: `AgroGrow/results/storage_model_comparison.png`
- Best regression binary: `AgroGrow/weights/best_storage_model.pkl`
