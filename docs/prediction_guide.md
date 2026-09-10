# AgroGrow Prediction & CLI Inference Guide

This guide explains how to run individual quality assessments on raw corn cobs using the command-line interface.

## Command Line Arguments

The `predict.py` script provides arguments to customize environmental variables and AI chat questions:

- `--image`: (Required) Path to the raw JPG image.
- `--temp`: (Default: `25.0`) Storage environment temperature in Celsius.
- `--humidity`: (Default: `70.0`) Relative humidity percentage.
- `--storage`: (Default: `"Open Air"`) Choice of storage style: `Open Air`, `Cold Storage`, or `Hermetic Bag`.
- `--query`: (Default: `"Is this batch suitable for export?"`) Specific question for the virtual agronomist.

---

## Example Commands

### 1. Test Regulated Storage Batch
Evaluate a corn cob using cold storage settings:
```bash
python predict.py --image "DataSet/data/test/corn/00GGXQ76763Y.jpg" --temp 8.0 --humidity 55.0 --storage "Cold Storage"
```

### 2. Custom AI Assistant Query
Inquire about variety and storage recommendations:
```bash
python predict.py --image "DataSet/data/test/corn/00GGXQ76763Y.jpg" --temp 12.0 --humidity 60.0 --storage "Cold Storage" --query "What variety is this and is it suitable for long-term storage?"
```

---

## Output Generated

Running the CLI script automatically produces:

1. **Terminal Summary Card**: Prints quality grade (A, B, C, D), confidence score, detected corn variety (powered by the MobileNetV3 CNN classifier), healthy/diseased/missing percentages, predicted shelf life (days), storage risk category, and the AI assistant's tailored answer.
2. **Visual Segmentation Overlay**: Saves a blended visualization overlay at `AgroGrow/results/{filename}_overlay.png` highlighting healthy kernels (green), missing sockets (blue), and diseased kernels (red).
3. **Quality Assessment PDF Report**: Generates a PDF report at `AgroGrow/reports/{filename}_quality_report.pdf` containing side-by-side images, metrics table, storage projections, and agronomist recommendations.
