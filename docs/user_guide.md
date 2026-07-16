# AgroGrow Streamlit Dashboard User Guide

This guide explains how to operate the interactive AgroGrow Streamlit dashboard to analyze corn cobs, review quality grading, predict storage shelf life, and download PDF reports.

## Launching the Dashboard

1. Activate your virtual environment:
   ```bash
   .\venv\Scripts\Activate.ps1
   ```
2. Run the Streamlit launcher in the project root:
   ```bash
   streamlit run AgroGrow/app.py
   ```
3. Your default web browser will automatically open to `http://localhost:8501`.

---

## Operating Instructions

### Step 1: Upload a Corn Image
Drag and drop or click to upload a raw post-harvest corn cob image in **JPG** or **JPEG** format. 
*Note: Samples from `DataSet/data/test/corn` can be used for testing.*

### Step 2: Configure Silo/Storage Conditions (Sidebar)
Before running the assessment, use the sliders and inputs in the sidebar to configure the storage profile:
- **Temperature Slider**: Set the ambient storage temperature (between 5.0°C and 45.0°C).
- **Humidity Slider**: Set the relative humidity in the silo (between 30% and 98% RH).
- **Storage Type**: Select the container type:
  - `Open Air`: Silos open to atmosphere (lower base shelf life).
  - `Cold Storage`: Thermally regulated systems (longer shelf life).
  - `Hermetic Bag`: Air-tight sealing bags (medium to long-term storage).

### Step 3: Run the AI Pipeline
Click the **"Run Quality Assessment Pipeline"** button. The dashboard will trigger the following processes:
1. **Semantic Segmentation**: The deep learning model (`Corn-Net`) segments the image into 4 classes.
2. **Overlay Mapping**: Renders a blended color visualization of the detections.
3. **Agronomic Scoring**: Computes Healthy %, Diseased %, Missing %, density, and overall score.
4. **Grading Classification**: Assigns Quality Grade A, B, C, or D.
5. **Shelf-Life Regression**: Compares conditions and predicts expected storage days and risk.

---

## Interpreting Dashboard Outputs

### KPI Panels
- **Quality Grade**: Displays Grade A, B, C, or D badge color-coded (Green for A, Blue for B, Orange for C, Red for D).
- **Quality Score**: Real-time score out of 100 representing raw kernel health.
- **Estimated Shelf Life**: Projected storage limit (in days) before significant decay.
- **Storage Risk Category**: Visual indicator of storage risk (Low, Medium, High).

### Visual Layout
- **Left**: Original uploaded image.
- **Right**: Color-coded segmented overlay:
  - 🟢 **Green**: Healthy kernel segments.
  - 🔴 **Red**: Missing kernels / gaps.
  - 🔵 **Blue**: Diseased kernel segments.

### AI Assistant Chat
Use the input chat box to interact with the virtual agronomist. Try typing:
- *"Why is this corn Grade B?"*
- *"What caused the quality reduction?"*
- *"Is this batch suitable for export?"*

### PDF Report Downloads
Click **"Compile Report PDF"** to build a printable PDF file, then click **"Download PDF Quality Report"** to save it locally.
