# AgroGrow — Installation Guide

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.9+ | 3.10+ |
| RAM | 8 GB | 16 GB |
| GPU | Optional (CUDA 11.6+) | NVIDIA with 4GB+ VRAM |
| Storage | 5 GB | 10 GB |
| OS | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/AgroGrow.git
cd "research paper 1/AgroGrow"
```

## Step 2: Create Virtual Environment

```bash
python -m venv ../venv
# Windows:
..\venv\Scripts\activate
# Linux/Mac:
source ../venv/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **PyTorch** (deep learning — CornNet segmentation + MobileNetV3 variety classifier)
- **Streamlit** (web dashboard)
- **OpenCV** (image processing)
- **XGBoost** (shelf-life regression)
- **google-generativeai** (Gemini AI chatbot)
- **icrawler** (variety dataset scraping)
- **ReportLab** (PDF report generation)

## Step 4: (Optional) Enable Gemini AI Chatbot

Get a **free** API key at https://aistudio.google.com

```powershell
# Windows PowerShell:
$env:GEMINI_API_KEY = "your_key_here"

# Linux/Mac:
export GEMINI_API_KEY=your_key_here
```

Without a key, CornBot runs in enhanced **rule-based mode** (still context-aware).

## Step 5: Run the Dashboard

```powershell
# From the "research paper 1" folder:
$env:PYTHONPATH = "d:\Sem_7\research paper 1"
.\venv\Scripts\streamlit run AgroGrow\app.py
```

Access at: **http://localhost:8501**

---

## Optional: Train the Variety Classifier (MobileNetV3)

The MobileNetV3 corn variety classifier improves variety identification beyond colour-grading.

### Step A — Download Variety Images (~5-10 mins, requires internet)
```bash
python variety_data/build_variety_dataset.py --images_per_class 200
```
Downloads ~200 images for each of: Dent, Flint, Sweet, Popcorn, Blue corn.

### Step B — Train Classifier (~15-20 mins on GPU)
```bash
python variety_data/train_variety_classifier.py --epochs 25 --batch_size 16
```
Saves `weights/variety_classifier.pth`. The app automatically uses it next run.

---

## Optional: Train the Segmentation Model (CornNet)

```bash
python train.py
```
Uses the 1,387-image labelled dataset in `dataset/`. Saves to `weights/best_model.pth`.

---

## Running Tests

```bash
python tests/run_tests.py
# Expected output: ALL 7 UNIT TESTS PASSED SUCCESSFULLY!
```

---

## Troubleshooting

| Error | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'AgroGrow'` | Set `PYTHONPATH` to the parent of `AgroGrow/` |
| `CUDA out of memory` | Select "CPU (Safe Mode)" in the sidebar |
| `Weights not found` | Run `train.py` first, or download pre-trained weights |
| `Gemini API error` | Check that `GEMINI_API_KEY` is set correctly |
| `No corn detected` | Upload a close-up JPG of a corn cob, not a field photo |
