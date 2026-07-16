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
st.sidebar.markdown("### 📂 Model Configurations")
device_detected = "GPU (CUDA)" if torch.cuda.is_available() else "CPU Fallback"
st.sidebar.info(f"**Inference Device:** {device_detected}")

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
    # Save file to a temporary directory
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tfile.write(uploaded_file.read())
    tfile.close()
    
    img_path = Path(tfile.name)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📷 Original Image")
        st.image(str(img_path), use_container_width=True)
        
    # Run Inference
    if st.button("🔍 Run Quality Assessment Pipeline", use_container_width=True):
        with st.spinner("Analyzing kernels, extracting features, and mapping storage life..."):
            try:
                # 1. Segmentation
                predictor = CornPredictor()
                mask, overlay, _, confidence = predictor.predict_single(img_path)
                
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
                    {"role": "assistant", "content": f"Hi! I am the AgroGrow expert assistant. I have reviewed this batch and it was classified as **{grading['grade']}** with a Quality Score of **{features['quality_score']:.1f}/100**. Ask me anything about this assessment."}
                ]
                
            except Exception as ex:
                st.error(f"Assessment Pipeline Error: {ex}")
                logger.exception(ex)

    # Display Results if Assessment is Cached
    if st.session_state.current_analysis is not None:
        res = st.session_state.current_analysis
        
        with col2:
            st.markdown("### 🎯 Classification Overlay Mapping")
            st.image(str(res["overlay_path"]), use_container_width=True)
            
            # Map legend
            st.markdown("""
                <div style='text-align: center; margin-top: 10px;'>
                    <span style='color:#00FF00; font-weight:bold; margin-right:15px;'>■ Healthy Kernels</span>
                    <span style='color:#FF0000; font-weight:bold; margin-right:15px;'>■ Missing Kernels</span>
                    <span style='color:#0000FF; font-weight:bold;'>■ Diseased Kernels</span>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
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
            
        st.markdown("Spacer")
        
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
            st.dataframe(feat_df, use_container_width=True, hide_index=True)
            
        with det_col2:
            st.markdown("#### 📊 Kernel Segmentation Ratios")
            chart_df = pd.DataFrame({
                "Class": ["Healthy Kernels", "Diseased Kernels", "Missing Kernels"],
                "Percentage (%)": [
                    res['features']['healthy_percentage'],
                    res['features']['disease_percentage'],
                    res['features']['missing_percentage']
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
                    "Diseased Kernels": "#4299E1",
                    "Missing Kernels": "#F56565"
                },
                text_auto=".2f"
            )
            fig.update_layout(showlegend=False, height=220, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
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
                        use_container_width=True
                    )
                    
        # Prediction History Log
        st.markdown("---")
        st.markdown("### 🕒 Assessment History Log (Current Session)")
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df, use_container_width=True)
        else:
            st.write("No uploads analyzed yet.")
