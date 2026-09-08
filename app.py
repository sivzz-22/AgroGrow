"""
AgroGrow Interactive Streamlit Dashboard.
Provides a premium web interface for upload, segmentation mapping,
agronomic quality analytics, shelf-life model predictions, and PDF report downloads.
"""

import os
import shutil
import tempfile
import sys
from pathlib import Path
import streamlit as st
import numpy as np
import cv2
import plotly.express as px
import pandas as pd
import torch

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from AgroGrow.config import global_config
from AgroGrow.utils.logger import logger
from AgroGrow.prediction.predictor import CornPredictor
from AgroGrow.prediction.feature_extractor import CornFeatureExtractor
from AgroGrow.prediction.grader import CornGrader
from AgroGrow.storage_prediction.model import StoragePredictor
from AgroGrow.assistant.agent import CornAssistant
from AgroGrow.utils.report_generator import PDFReportGenerator

# 1. Page Configuration
st.set_page_config(
    page_title="AgroGrow — Corn Quality Assessment",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom Styling CSS Injection
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        .main {
            background-color: #F8FAFC;
        }
        
        /* Metric cards styling */
        .metric-card {
            background-color: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
            border-left: 5px solid #2B6CB0;
            text-align: center;
        }
        
        .metric-value {
            font-size: 28px;
            font-weight: 800;
            color: #1A365D;
            margin-top: 5px;
        }
        
        .metric-title {
            font-size: 13px;
            color: #718096;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        
        /* Badges */
        .badge {
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 12px;
            display: inline-block;
        }
        .badge-green { background-color: #DEF7EC; color: #03543F; }
        .badge-blue { background-color: #E1EFFE; color: #1E429F; }
        .badge-orange { background-color: #FDF6B2; color: #723B13; }
        .badge-red { background-color: #FDE8E8; color: #9B1C1C; }
    </style>
""", unsafe_allow_html=True)

# Initialize Session States
if "history" not in st.session_state:
    st.session_state.history = []
if "assistant_messages" not in st.session_state:
    st.session_state.assistant_messages = []
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None

# Sidebar Controls
st.sidebar.markdown("<h2 style='color:#1A365D;'>🌽 AgroGrow Controls</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown("### 🌡️ Storage Parameters")
temp_input = st.sidebar.slider("Temperature (°C)", min_value=5.0, max_value=45.0, value=25.0, step=0.5)
humidity_input = st.sidebar.slider("Humidity (%)", min_value=30.0, max_value=98.0, value=70.0, step=1.0)
storage_type = st.sidebar.selectbox("Storage Type", global_config.storage_types)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌾 Grain Variety")
corn_variety_input = st.sidebar.selectbox(
    "Corn Variety Mode",
    options=[
        "🔍 Auto-Detect Variety",
        "🌾 Indian / Flint / Multi-colored Corn (Ruby / Bronze / Purple)",
        "🌽 Commercial Dent Corn (Yellow/White)"
    ],
    index=0,
    help="For multicolored or ruby-red flint corn, ensures natural grain pigmentation is graded as healthy, not defect."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Display Theme")
overlay_style = st.sidebar.radio(
    "Overlay Background",
    options=[
        "🌫️ Studio Light Neutral (Clean & Modern)",
        "🌤️ Soft Light Translucent Blend (Non-Opaque)",
        "🖼️ Semi-Transparent Photo Blend",
        "🌑 Dark Slate / Black"
    ],
    index=0,
    help="Studio Light Neutral replaces harsh black with a clean studio backdrop. Soft Light provides a gentle translucent wash."
)
if "Studio Light" in overlay_style:
    bg_style_choice = "light"
    bg_legend_desc = "Background outside cob (Studio Light Slate)"
    bg_legend_color = "#E2E8F0"
elif "Soft Light" in overlay_style:
    bg_style_choice = "soft_blend"
    bg_legend_desc = "Background outside cob (Translucent Light Veil)"
    bg_legend_color = "#CBD5E1"
elif "Photo Blend" in overlay_style:
    bg_style_choice = "photo"
    bg_legend_desc = "Background outside cob (Natural Photo Context)"
    bg_legend_color = "#94A3B8"
else:
    bg_style_choice = "black"
    bg_legend_desc = "Background outside cob (Dark Slate / Black)"
    bg_legend_color = "#1E293B"

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Model Configurations")
has_cuda = torch.cuda.is_available()
device_options = ["Auto (GPU if available)", "CPU (Safe Mode)"] if has_cuda else ["CPU (Safe Mode)"]
device_choice = st.sidebar.selectbox("Inference Hardware", device_options, index=0)
use_device = "cpu" if "CPU" in device_choice else None
st.sidebar.info(f"**Active Mode:** {'GPU Accelerated' if (has_cuda and use_device != 'cpu') else 'CPU Safe Mode'}")

# Check model weights
weights_file = global_config.weights_dir / "best_model.pth"
if weights_file.exists():
    st.sidebar.success("Corn-Net Model Weights Loaded")
else:
    st.sidebar.warning("Using Initialized Random Weights (Please train first)")

# Header Logo
st.markdown("<h1 style='color:#1A365D; text-align:center;'>🌽 AgroGrow Corn Quality Assessment System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#4A5568;'>AI-Powered semantic segmentation quality grading and post-harvest shelf-life forecasting.</p>", unsafe_allow_html=True)
st.markdown("---")

# File Upload Section
uploaded_file = st.file_uploader("Upload Raw Corn Image (.jpg)", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tfile.write(uploaded_file.getvalue())
    tfile.close()
    img_path = Path(tfile.name)

    # Load image + run auto-crop immediately on upload
    from AgroGrow.prediction.predictor import auto_crop_corn_ear
    _raw_bgr = cv2.imread(str(img_path))
    _raw_rgb = cv2.cvtColor(_raw_bgr, cv2.COLOR_BGR2RGB)
    _H, _W   = _raw_rgb.shape[:2]
    _auto_cropped, _auto_bbox = auto_crop_corn_ear(_raw_rgb)
    _ax, _ay, _aw, _ah = _auto_bbox
    _is_full = (_aw >= _W * 0.95 and _ah >= _H * 0.95)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📷 Original Image")
        st.image(_raw_rgb, width='stretch')

    with col2:
        st.markdown("### ✂️ Image Preparation")

        # ── One-click crop mode selector ──────────────────────
        crop_mode = st.radio(
            "Choose how to send the image to the model:",
            options=["🌽 Auto Crop", "🖐 Manual Crop", "📷 Full Image"],
            index=0,
            horizontal=True
        )

        if crop_mode == "🌽 Auto Crop":
            if _is_full and _W > _H * 1.1:
                st.info("🌾 Multi-ear / Wide image detected — auto-focusing on prominent foreground cob body.")
                _ac_x1 = int(_W * 0.10)
                _ac_x2 = int(_W * 0.50)
                _ac_cropped = _raw_rgb[:, _ac_x1:_ac_x2]
            else:
                _ac_cropped = _auto_cropped
                if _is_full:
                    st.info("ℹ️ Using full frame for inference.")
                else:
                    st.success(f"✅ Corn ear auto-detected and cropped ({_aw}×{_ah} px). Background removed.")
            st.image(_ac_cropped, width='stretch',
                     caption="This region will be analysed by the model")
            _c = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(_c.name, cv2.cvtColor(_ac_cropped, cv2.COLOR_RGB2BGR))
            _c.close()
            inference_img_path = Path(_c.name)

        elif crop_mode == "🖐 Manual Crop":
            st.info("💡 Adjust boundaries or click a quick-framing preset below:")
            p1, p2, p3, p4 = st.columns(4)
            if p1.button("🎯 Center Cob", key="btn_center"):
                st.session_state["mc_x1"] = int(_W * 0.25)
                st.session_state["mc_x2"] = int(_W * 0.48)
                st.session_state["mc_y1"] = int(_H * 0.05)
                st.session_state["mc_y2"] = int(_H * 0.95)
            if p2.button("🌽 Left Cob", key="btn_left"):
                st.session_state["mc_x1"] = int(_W * 0.10)
                st.session_state["mc_x2"] = int(_W * 0.35)
                st.session_state["mc_y1"] = int(_H * 0.05)
                st.session_state["mc_y2"] = int(_H * 0.95)
            if p3.button("🌾 Both Cobs", key="btn_both"):
                st.session_state["mc_x1"] = int(_W * 0.10)
                st.session_state["mc_x2"] = int(_W * 0.50)
                st.session_state["mc_y1"] = int(_H * 0.05)
                st.session_state["mc_y2"] = int(_H * 0.95)
            if p4.button("🔄 Reset Full", key="btn_full"):
                st.session_state["mc_x1"] = 0
                st.session_state["mc_x2"] = _W
                st.session_state["mc_y1"] = 0
                st.session_state["mc_y2"] = _H

            init_x1 = st.session_state.get("mc_x1", int(_W * 0.22) if _is_full else max(0, _ax))
            init_x2 = st.session_state.get("mc_x2", int(_W * 0.49) if _is_full else min(_W, _ax + _aw))
            init_y1 = st.session_state.get("mc_y1", int(_H * 0.05) if _is_full else max(0, _ay))
            init_y2 = st.session_state.get("mc_y2", int(_H * 0.95) if _is_full else min(_H, _ay + _ah))

            cx1 = st.slider("Left Boundary", 0, _W - 10, init_x1, key="slider_x1")
            cx2 = st.slider("Right Boundary", cx1 + 10, _W, max(cx1 + 10, init_x2), key="slider_x2")
            cy1 = st.slider("Top Boundary", 0, _H - 10, init_y1, key="slider_y1")
            cy2 = st.slider("Bottom Boundary", cy1 + 10, _H, max(cy1 + 10, init_y2), key="slider_y2")

            _mc = _raw_rgb[cy1:cy2, cx1:cx2]
            st.image(_mc, width='stretch',
                     caption=f"Framed Cob Region — {cx2-cx1}×{cy2-cy1} px")
            _c = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(_c.name, cv2.cvtColor(_mc, cv2.COLOR_RGB2BGR))
            _c.close()
            inference_img_path = Path(_c.name)

        else:  # Full Image
            st.info("📷 Full image will be used. Outdoor background & husk suppression will be applied.")
            st.image(_raw_rgb, width='stretch')
            inference_img_path = img_path

    # Run Inference
    if st.button("🔍 Run Quality Assessment Pipeline", width='stretch'):
        with st.spinner("Analyzing kernels, extracting features, and mapping storage life..."):
            try:
                # 1. Segmentation (uses cropped or full image prepared by user)
                predictor = CornPredictor(device=use_device)
                mask, overlay, _, confidence = predictor.predict_single(
                    inference_img_path, auto_crop=False, corn_variety=corn_variety_input, background_style=bg_style_choice
                )

                
                # Save overlay temporarily for plotting
                overlay_temp_path = Path(tempfile.gettempdir()) / f"{img_path.stem}_overlay.png"
                cv2.imwrite(str(overlay_temp_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                
                # 2. Features
                features = CornFeatureExtractor.extract_features(mask)
                
                # 3. Grading
                grading = CornGrader.classify_grade(features, confidence)
                
                # 4. Storage Life
                try:
                    storage_predictor = StoragePredictor()
                    shelf_life, risk_level, recommendation = storage_predictor.predict(
                        healthy_pct=features["healthy_percentage"],
                        disease_pct=features["disease_percentage"],
                        missing_pct=features["missing_percentage"],
                        defect_area=features["defect_area_pixels"],
                        corn_area=features["total_corn_area_pixels"],
                        temperature=temp_input,
                        humidity=humidity_input,
                        storage_type=storage_type
                    )
                except Exception as e:
                    logger.warning(f"Failed loading storage prediction model: {e}. Running default heuristic.")
                    # Heuristic calculation fallback
                    shelf_life = 45.0
                    risk_level = "Medium Risk"
                    recommendation = "Default: Aerate grain and store in cool, low moisture silo."
                    
                # Cache results in Session State
                analysis_results = {
                    "image_path": img_path,
                    "overlay_path": overlay_temp_path,
                    "features": features,
                    "grading": grading,
                    "variety": getattr(predictor, "last_detected_variety", "Standard Dent Corn (Yellow/White)"),
                    "storage": {
                        "shelf_life_days": shelf_life,
                        "risk_level": risk_level,
                        "recommendation": recommendation,
                        "temperature_c": temp_input,
                        "humidity_pct": humidity_input,
                        "storage_type": storage_type
                    }
                }
                st.session_state.current_analysis = analysis_results
                
                # Add to history log
                st.session_state.history.append({
                    "Filename": uploaded_file.name,
                    "Grade": grading["grade"],
                    "Quality Score": f"{features['quality_score']:.1f}/100",
                    "Shelf Life": f"{shelf_life:.1f} Days",
                    "Risk": risk_level
                })
                
                # Initialize AI assistant messages
                st.session_state.assistant_messages = [
                    {"role": "assistant", "content": f"Hi! I am the AgroGrow expert assistant. I have reviewed this batch ({analysis_results['variety']}) and it was classified as **{grading['grade']}** with a Quality Score of **{features['quality_score']:.1f}/100**. Ask me anything about this assessment."}
                ]
                
            except Exception as ex:
                st.error(f"Assessment Pipeline Error: {ex}")
                logger.exception(ex)

    # Display Results if Assessment is Cached
    if st.session_state.current_analysis is not None:
        res = st.session_state.current_analysis

        st.markdown("---")
        # Variety Banner
        variety_name = res.get("variety", "Standard Dent Corn (Yellow/White)")
        if "flint" in variety_name.lower() or "indian" in variety_name.lower():
            st.markdown(f"""
                <div style='background:#F0FDF4; border-left:4px solid #16A34A; padding:12px 16px; border-radius:8px; margin-bottom:14px;'>
                    <b style='color:#15803D;'>🌾 Identified Variety:</b> <span style='font-size:15px; color:#166534;'><b>{variety_name}</b></span><br/>
                    <small style='color:#15803D;'>Naturally pigmented ruby-red and purple anthocyanin grains are correctly recognized as healthy culinary grain, not disease defects.</small>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🎯 Classification Overlay Mapping")
        ov_col1, ov_col2 = st.columns([1, 1])
        with ov_col1:
            st.image(str(res["overlay_path"]), width='stretch',
                     caption="Segmentation overlay — model output")
        with ov_col2:
            st.markdown(f"""
                <div style='padding:20px; background:#F8FAFC; border-radius:12px; border:1px solid #E2E8F0;'>
                <h4 style='color:#1A365D; margin-top:0;'>Colour Legend</h4>
                <p><span style='color:#00BB00; font-size:22px; font-weight:bold;'>■</span>
                   &nbsp;<b>Healthy Kernels (Green)</b></p>
                <p><span style='color:#0066FF; font-size:22px; font-weight:bold;'>■</span>
                   &nbsp;<b>Missing Kernel Sockets (Blue)</b></p>
                <p><span style='color:#FF0000; font-size:22px; font-weight:bold;'>■</span>
                   &nbsp;<b>Diseased / Rotten Kernels (Red)</b></p>
                <p><span style='color:{bg_legend_color}; font-size:22px; font-weight:bold; text-shadow: 0 0 1px #94A3B8;'>■</span>
                   &nbsp;<b>{bg_legend_desc}</b></p>
                </div>
            """, unsafe_allow_html=True)


        st.markdown("### 📊 Metrics Summary")
        
        # Dashboard KPI Cards
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        grade_text = res["grading"]["grade"]
        badge_style = "badge-green"
        if "Grade B" in grade_text:
            badge_style = "badge-blue"
        elif "Grade C" in grade_text:
            badge_style = "badge-orange"
        elif "Grade D" in grade_text:
            badge_style = "badge-red"
            
        with kpi_col1:
            st.markdown(f"""
                <div class="metric-card" style="border-left-color: #4299E1;">
                    <div class="metric-title">Quality Grade</div>
                    <div class="metric-value">{grade_text}</div>
                    <span class="badge {badge_style}">Confidence: {res['grading']['confidence']:.1f}%</span>
                </div>
            """, unsafe_allow_html=True)
            
        with kpi_col2:
            st.markdown(f"""
                <div class="metric-card" style="border-left-color: #48BB78;">
                    <div class="metric-title">Quality Score</div>
                    <div class="metric-value">{res['features']['quality_score']:.1f}/100</div>
                    <div style="font-size:12px; color:#718096; margin-top:4px;">Healthy: {res['features']['healthy_percentage']:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
        with kpi_col3:
            st.markdown(f"""
                <div class="metric-card" style="border-left-color: #ED8936;">
                    <div class="metric-title">Estimated Shelf Life</div>
                    <div class="metric-value">{res['storage']['shelf_life_days']:.1f} Days</div>
                    <div style="font-size:12px; color:#718096; margin-top:4px;">At {res['storage']['temperature_c']}°C, {res['storage']['humidity_pct']}% RH</div>
                </div>
            """, unsafe_allow_html=True)
            
        risk_color = "#38A169" if "Low" in res['storage']['risk_level'] else ("#DD6B20" if "Medium" in res['storage']['risk_level'] else "#E53E3E")
        with kpi_col4:
            st.markdown(f"""
                <div class="metric-card" style="border-left-color: {risk_color};">
                    <div class="metric-title">Storage Risk Category</div>
                    <div class="metric-value" style="color: {risk_color};">{res['storage']['risk_level']}</div>
                    <div style="font-size:12px; color:#718096; margin-top:4px;">Type: {res['storage']['storage_type']}</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Detail Panel (Table + Chart side by side)
        det_col1, det_col2 = st.columns([1, 1.2])
        
        with det_col1:
            st.markdown("#### 📋 Agronomic Features Details")
            feat_df = pd.DataFrame({
                "Metric": [
                    "Healthy Kernels Percentage",
                    "Diseased Kernels Percentage",
                    "Missing Kernels Percentage",
                    "Total Corn area (pixels)",
                    "Defect area (pixels)",
                    "Kernel Density Blobs Count",
                    "Bounding Box Width/Height"
                ],
                "Value": [
                    f"{res['features']['healthy_percentage']:.2f}%",
                    f"{res['features']['disease_percentage']:.2f}%",
                    f"{res['features']['missing_percentage']:.2f}%",
                    f"{res['features']['total_corn_area_pixels']} px",
                    f"{res['features']['defect_area_pixels']} px",
                    f"{res['features']['num_healthy_blobs']} units",
                    f"{res['features']['bounding_box'][2]}x{res['features']['bounding_box'][3]}"
                ]
            })
            st.dataframe(feat_df, width='stretch', hide_index=True)
            
        with det_col2:
            st.markdown("#### 📊 Kernel Segmentation Ratios")
            chart_df = pd.DataFrame({
                "Class": ["Healthy Kernels", "Missing Kernels", "Diseased Kernels"],
                "Percentage (%)": [
                    res['features']['healthy_percentage'],
                    res['features']['missing_percentage'],
                    res['features']['disease_percentage']
                ]
            })
            fig = px.bar(
                chart_df,
                x="Percentage (%)",
                y="Class",
                orientation="h",
                color="Class",
                color_discrete_map={
                    "Healthy Kernels": "#48BB78",
                    "Missing Kernels": "#4299E1",
                    "Diseased Kernels": "#F56565"
                },
                text_auto=".2f"
            )
            fig.update_layout(showlegend=False, height=220, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig, width='stretch')
            
        st.markdown("---")
        
        # Dynamic AI Assistant Section & PDF download side by side
        ass_col1, ass_col2 = st.columns([1.2, 1])
        
        with ass_col1:
            st.markdown("### 💬 AI Agronomist Assistant")
            
            # Display chat messages
            for msg in st.session_state.assistant_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
            # User chat input
            user_msg = st.chat_input("Ask a question about this corn batch (e.g. 'Why is this corn Grade B?')")
            if user_msg:
                with st.chat_message("user"):
                    st.markdown(user_msg)
                st.session_state.assistant_messages.append({"role": "user", "content": user_msg})
                
                # Run assistant logic
                assistant = CornAssistant()
                
                # Package current details into context
                context = {
                    **res["features"],
                    **res["grading"],
                    **res["storage"]
                }
                
                with st.chat_message("assistant"):
                    with st.spinner("Generating expert analysis..."):
                        reply = assistant.answer_query(user_msg, context)
                        st.markdown(reply)
                st.session_state.assistant_messages.append({"role": "assistant", "content": reply})
                
        with ass_col2:
            st.markdown("### 📄 Quality Assessment PDF Report")
            st.info("Compile a premium, formatted PDF report containing all segmentation overlays, tabular statistics, shelf-life model predictions, and AI recommendations.")
            
            # Setup PDF output path
            report_name = f"{img_path.stem}_quality_report.pdf"
            pdf_path = global_config.reports_dir / report_name
            
            # Get latest assistant reply as summary text
            summary_recommendation = st.session_state.assistant_messages[0]["content"]
            if len(st.session_state.assistant_messages) > 1:
                # Find latest assistant text
                for m in reversed(st.session_state.assistant_messages):
                    if m["role"] == "assistant":
                        summary_recommendation = m["content"]
                        break
                        
            # Compile PDF Button
            if st.button("🛠️ Compile Report PDF"):
                with st.spinner("Generating PDF layout via ReportLab..."):
                    try:
                        PDFReportGenerator.generate(
                            image_path=res["image_path"],
                            overlay_path=res["overlay_path"],
                            features=res["features"],
                            grading=res["grading"],
                            storage=res["storage"],
                            assistant_text=summary_recommendation,
                            output_pdf_path=pdf_path
                        )
                        st.success(f"PDF Successfully compiled: {report_name}")
                    except Exception as e:
                        st.error(f"ReportLab failed: {e}")
                        
            # Download Button (only if file exists)
            if pdf_path.exists():
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download PDF Quality Report",
                        data=pdf_file,
                        file_name=report_name,
                        mime="application/pdf",
                        width='stretch'
                    )
                    
        # Prediction History Log
        st.markdown("---")
        st.markdown("### 🕒 Assessment History Log (Current Session)")
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df, width='stretch')
        else:
            st.write("No uploads analyzed yet.")
