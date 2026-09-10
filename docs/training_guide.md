# AgroGrow Deep Learning Training Guide

This guide details how to prepare datasets, run deep learning semantic segmentation training, train the multi-variety CNN classifier, and execute storage shelf-life regression training.

---

## 1. Segmentation Pipeline (CornNet — PyTorch)

### Step A: Setup Preprocessing and Validation
Execute the preprocessor to generate synthetic HSV/defect masks (healthy, missing, diseased classes) and validate split balance:
```bash
python AgroGrow/utils/prepare_dataset.py
```
*Verification*: Check `AgroGrow/reports/dataset_validation_report.md` to confirm the splits compiled successfully.

### Step B: Launch Training
Execute the training pipeline for Corn-Net (U-Net with ResNet encoder):
```bash
python AgroGrow/train.py
```
*Options and Defaults (Configurable in `AgroGrow/config.py`):*
- **Batch Size**: 8
- **Learning Rate**: 1e-4
- **Epochs**: 15 (with Early Stopping threshold of 5 epochs)
- **Model Output**: Saves best parameters dynamically to `AgroGrow/weights/best_model.pth`.

### Step C: Check Training Curves
Open `AgroGrow/results/training_metrics_curves.png` to examine training/validation loss, Mean IoU (mIoU), and Mean Dice score across training cycles.

---

## 2. Corn Variety Classification Pipeline (MobileNetV3 — PyTorch)

AgroGrow includes a lightweight MobileNetV3 convolutional neural network for classifying raw corn cobs into **5 global grain varieties**:
1. **Commercial Dent Corn** (*Zea mays indentata*) — Yellow/White
2. **Indian / Multi-Colored Flint Corn** (*Zea mays indurata*) — Ruby, purple, bronze anthocyanins
3. **Sweet Corn** (*Zea mays saccharata*) — High sugar, pale cream
4. **Popcorn** (*Zea mays everta*) — Small hard spherical kernels
5. **Blue / Black Corn** (*Zea mays* — Hopi Blue) — Deep blue/black anthocyanins

### Step A: Build the Variety Dataset
Synthesize multi-variety dataset crops with realistic kernel texture, anthocyanin pigmentation, and defect variations:
```bash
python AgroGrow/variety_data/build_variety_dataset.py
```
*Dataset outputs*: `AgroGrow/variety_data/splits/{train,val}/` partitioned by class.

### Step B: Train MobileNetV3 Classifier
Train the classifier using a two-phase transfer learning schedule:
- **Phase 1 (Epochs 1–9)**: Feature backbone frozen; trains head layers with Adam (lr=1e-3).
- **Phase 2 (Epochs 10–25)**: Full network fine-tuning with Cosine Annealing (lr=1e-4).
```bash
python AgroGrow/variety_data/train_variety_classifier.py
```
*Outputs*:
- Trained weights saved to: `AgroGrow/weights/variety_classifier.pth` (~5.9 MB, **>90% validation accuracy**).
- Model is automatically detected by `predictor.py` at runtime without restarting the app.

---

## 3. Storage Life Prediction Pipeline (Scikit-Learn & XGBoost)

### Step A: Synthesize Storage Environmental Database
Run the simulator to construct a database containing environmental variables and storage targets:
```bash
python AgroGrow/storage_prediction/dataset_generator.py
```
*Generates*: `AgroGrow/dataset/storage_data.csv` containing 1,200 simulated warehouse logs across Open Air, Cold Storage, and Hermetic Bag conditions.

### Step B: Train Regressors & Compare
Fit Random Forest and XGBoost regressors, compare R² and RMSE, and serialize the champion model:
```bash
python -c "from AgroGrow.storage_prediction.model import train_storage_models; train_storage_models()"
```
*Outputs*:
- Model comparison chart: `AgroGrow/results/storage_model_comparison.png`
- Best regression model: `AgroGrow/weights/best_storage_model.pkl`
