# AgroGrow Demonstration Samples

This directory contains curated, model-tested corn images specifically prepared for live demonstrations and presentations.

| File | Category / Scenario | What it Demonstrates | Expected App Output |
| :--- | :--- | :--- | :--- |
| `01_healthy_corn_grade_A.jpg` | **Prime Commercial Corn** | Intact kernels, disease-free ear | **Grade A**, High Quality Score (95–100%), Green Health Banner |
| `02_missing_kernels_corn.jpg` | **Physical Defect / Harvest Loss** | Empty kernel sockets along cob | **Grade B/C**, Blue socket mask overlay, Moderate Defects Banner |
| `03_severe_diseased_corn.jpg` | **Fungal Rot / Disease** | Severe mould and tissue necrosis | **Grade D**, Red disease mask, Critical Warning Banner, Short Shelf Life |
| `04_indian_flint_multicolor.jpg` | **Heirloom Variety (Flint)** | Multi-coloured kernels | **Indian/Flint Corn** auto-detected (99.8% conf.), pigments preserved |
| `05_hopi_blue_black_corn.jpg` | **Pigmented Variety (Blue/Black)** | Dark anthocyanin pigmentation | **Hopi Blue Corn** auto-detected (93.5% conf.), prevents false disease alarm |
| `06_non_corn_foliage.jpg` | **Negative Control (No Corn)** | Non-corn background image | **"No Corn Detected in Image"** notification banner |

### How to use during demonstration:
1. Open the AgroGrow dashboard in your browser.
2. Drag and drop any image from this `demo_samples` folder into the file uploader.
3. Click **"Run Quality Assessment Pipeline"**.
4. Show the visual overlay, variety badge, USDA grading, storage shelf-life card, and ask **CornBot** questions!
