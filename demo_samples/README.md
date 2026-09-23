# AgroGrow Demonstration Samples

This directory contains **model-verified** corn images specifically tested and curated for live demonstrations and conference presentations.

> All images below have been run through the full AgroGrow pipeline and produce correct, meaningful outputs.

| File | Category / Scenario | Model Output — What You Will See |
| :--- | :--- | :--- |
| `01_healthy_corn_grade_A.jpg` | ✅ **Prime Healthy Corn** | Grade A · Score 100/100 · 100% Healthy · Green Banner · "Corn Batch is Healthy" |
| `02_missing_kernels_corn.jpg` | 🔵 **Missing Kernels Defect** | Grade C · Score 67/100 · 84% Healthy · 15% Missing · Moderate Defects Banner |
| `03_severe_diseased_corn.jpg` | 🔴 **Severe Disease / Rot** | Grade D · Score 0/100 · 97% Diseased · Red Disease Overlay · Critical Alert Banner |

### How to use for demonstration:
1. Open the AgroGrow dashboard at **http://localhost:8501**
2. Drag and drop any image from this folder into the **file uploader**.
3. Leave framing as **"🌽 Auto Crop Ear"** (default).
4. Click **"🔍 Run Quality Assessment Pipeline"**.
5. Show the audience:
   - 🎨 **Segmentation Overlay** with colour-coded kernel mask
   - 📊 **Quality Metrics** (4 KPI cards)
   - 🏭 **Storage Analysis** grid
   - 📋 **Grade Summary** and **💡 Storage Recommendation**
   - 💬 Ask **AgroGrow AI Chatbot** questions about the result

### Suggested Demo Script:
1. **Image 01** → Show a Grade A perfect result → *"The system detects 100% healthy kernels and recommends long-term hermetic storage"*
2. **Image 02** → Show Grade C with missing sockets → *"Blue regions indicate empty sockets — kernel loss from harvesting"*
3. **Image 03** → Show Grade D critical disease → *"The AI flags fungal rot immediately and recommends emergency aeration"*
