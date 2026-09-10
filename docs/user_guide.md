# AgroGrow Streamlit Dashboard User Guide

This guide explains how to operate the interactive AgroGrow Streamlit dashboard to analyze corn cobs, review quality grading, detect grain varieties, predict storage shelf life, interact with the AI chatbot, and download PDF reports.

---

## Launching the Dashboard

1. Activate your virtual environment:
   ```bash
   ..\venv\Scripts\Activate.ps1
   ```
2. Run the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```
3. Your default web browser will automatically open to `http://localhost:8501`.

---

## Operating Instructions

### Step 1: Upload a Corn Image
Drag and drop or click to upload a raw post-harvest corn cob image in **JPG**, **JPEG**, or **PNG** format.
*Note: Samples from `DataSet/data/test/corn` can be used directly.*

### Step 2: Configure Silo/Storage Conditions (Sidebar)
Before running the assessment, customize your environmental variables in the sidebar:
- **Storage Type**: Choose between:
  - `Cold Storage`: Regulated sub-zero cold room or refrigerated warehouse (temperature slider expands from **-10.0°C to +15.0°C**).
  - `Open Air`: Ambient atmospheric silos (**10.0°C to 45.0°C**).
  - `Hermetic Bag`: Air-tight sealed storage bags (**10.0°C to 40.0°C**).
- **Storage Temperature Slider**: Dynamically adjusts based on chosen storage type (supporting true freezing / cold storage temperatures).
- **Relative Humidity Slider**: Adjust humidity in the silo (from 30% to 98% RH).
- **Corn Variety Selection**: Defaults to `Auto-Detect (MobileNet CNN)`. You can also manually choose between Dent, Flint, Sweet, Popcorn, or Blue/Black corn.

### Step 3: Run the AI Pipeline
Click the **"Run Quality Assessment Pipeline"** button. The system performs:
1. **Cob Auto-Cropping**: Strips away outdoor field background, leaves, soil, and sky.
2. **Grain Variety Detection**: Evaluates kernels via the fine-tuned **MobileNetV3 CNN** (>90% accuracy).
3. **Semantic Kernel Segmentation**: Deep neural network (`CornNet`) segments healthy, missing, and diseased regions.
4. **Photographic Blending**: Produces a clean, natural overlay without harsh opaque artifacts.
5. **Agronomic Grading**: Classifies into Grades A, B, C, or D with high precision.
6. **Shelf-Life Prediction**: Computes projected safe storage days and risk category.

---

## Interpreting Dashboard Outputs

### 1. Health Status & Intimation Banner
At the top of the assessment results, an intuitive status card intimates harvest health:
- 🟢 **Healthy Harvest**: Flagged when healthy kernels exceed 85% and diseased kernels are under 5%.
- 🟡 **Quality Warning**: Flagged when moderate defects or disease symptoms are detected.
- 🔴 **Quality Alert / Critical Defects**: Triggered when severe disease or large missing gaps threaten post-harvest life.
- ⚠️ **No Corn Detected**: If an uploaded photo contains no corn or an unreadable object, the banner explicitly warns that no corn is present.

### 2. Grain Variety Badge
Displays the detected global corn variety:
- **Commercial Dent Corn** (*Zea mays indentata*)
- **Indian / Multi-Colored Flint Corn** (*Zea mays indurata*)
- **Sweet Corn** (*Zea mays saccharata*)
- **Popcorn** (*Zea mays everta*)
- **Blue / Black Corn** (*Zea mays* — Hopi Blue)

### 3. KPI Scorecards
- **Quality Grade**: Badges for Grade A (Premium), Grade B (Good), Grade C (Fair), or Grade D (Rejected).
- **Quality Score**: Agronomic metric out of 100.
- **Estimated Shelf Life**: Safe storage duration in days.
- **Storage Risk Category**: Low, Medium, or High Risk badge.

### 4. Visual Inspection Layout
- **Left**: Original cropped raw image.
- **Right**: Photographic overlay with natural transparency:
  - 🟢 **Green**: Healthy kernels.
  - 🔵 **Blue**: Missing kernel gaps / sockets.
  - 🔴 **Red**: Diseased or damaged kernels.

### 5. Floating AI Agronomist Chatbot
At the bottom right of the screen, click the **AI Agronomist Chat** button to chat with **CornBot**:
- Ask questions about the analyzed cob:
  - *"Why did this batch receive Grade B?"*
  - *"What are the best storage settings for this variety?"*
  - *"Can this corn be stored for more than 6 months?"*
- Ask general corn botany questions:
  - *"Tell me about Indian Flint corn and its colors."*
  - *"How does Popcorn explode when heated?"*
  - *"What causes Fusarium ear rot?"*

### 6. PDF Report Download
Click **"Download Assessment PDF"** to save an agronomist-grade quality certificate locally.
