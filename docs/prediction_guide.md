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

### 1. Test Premium Batch
Evaluate a corn cob using regulated storage settings:
```bash
python AgroGrow/predict.py --image "AgroGrow/dataset/images/test/0NSA9HSJ4NTE.jpg" --temp 8.0 --humidity 55.0 --storage "Cold Storage"
```

### 2. Custom AI Assistant Query
Inquire about classification details:
```bash
python AgroGrow/predict.py --image "AgroGrow/dataset/images/test/0NSA9HSJ4NTE.jpg" --temp 32.0 --humidity 85.0 --storage "Open Air" --query "Why is this corn Grade B?"
```

---

## Output Files Created

Running the CLI script automatically generates several output artifacts:

1. **Terminal stdout**: Prints a clean structured summary card containing quality score, classification grade, predicted shelf life (days), risk status, and the AI assistant's text response.
2. **Visual Segmentation Overlay**: Saves a colored visualization overlay at `AgroGrow/results/{filename}_overlay.png` where colors indicate healthy (green), missing (red), and diseased (blue) kernels.
3. **Quality Assessment PDF Report**: Standard ReportLab report including the metadata, original vs segment side-by-side images, crop metrics table, storage projections, and the AI agronomist's written summary. Saved at `AgroGrow/reports/{filename}_quality_report.pdf`.
