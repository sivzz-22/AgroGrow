"""
AgroGrow Interactive Streamlit Dashboard — Premium Edition.
Clean, animated UI with health status banner, no-corn detection,
and a floating Gemini-powered chatbot.
"""

import tempfile
import sys
from pathlib import Path
import streamlit as st
import numpy as np
import cv2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).resolve().parent.parent))

from AgroGrow.config import global_config
from AgroGrow.utils.logger import logger
from AgroGrow.prediction.predictor import CornPredictor
from AgroGrow.prediction.feature_extractor import CornFeatureExtractor
from AgroGrow.prediction.grader import CornGrader
from AgroGrow.storage_prediction.model import StoragePredictor
from AgroGrow.utils.report_generator import PDFReportGenerator
from AgroGrow.assistant.chatbot import CornChatbot

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroGrow — Corn Quality Assessment",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Animated gradient header ── */
.ag-header {
    background: linear-gradient(135deg, #0f4c2a 0%, #1a6b3a 40%, #0d5c3e 70%, #1a4a2a 100%);
    background-size: 300% 300%;
    animation: gradientShift 8s ease infinite;
    border-radius: 20px;
    padding: 36px 32px 32px 32px;
    margin-bottom: 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.ag-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.04) 0%, transparent 60%);
    animation: rotate 15s linear infinite;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes rotate {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
.ag-title {
    font-size: 40px;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 12px rgba(0,0,0,0.3);
}
.ag-subtitle {
    color: rgba(255,255,255,0.8);
    font-size: 15px;
    margin: 8px 0 0 0;
    font-weight: 400;
}
.ag-pill {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 999px;
    padding: 4px 16px;
    font-size: 12px;
    color: rgba(255,255,255,0.9);
    font-weight: 600;
    margin-top: 14px;
    letter-spacing: 1px;
    text-transform: uppercase;
    backdrop-filter: blur(8px);
}

/* ── Health status banner ── */
.health-banner {
    border-radius: 14px;
    padding: 18px 24px;
    margin: 16px 0;
    display: flex;
    align-items: center;
    gap: 16px;
    animation: slideDown 0.4s ease;
}
@keyframes slideDown {
    from { opacity: 0; transform: translateY(-12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.health-banner-healthy {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1.5px solid #16a34a;
    box-shadow: 0 4px 20px rgba(22, 163, 74, 0.12);
}
.health-banner-warning {
    background: linear-gradient(135deg, #fffbeb, #fef9c3);
    border: 1.5px solid #d97706;
    box-shadow: 0 4px 20px rgba(217, 119, 6, 0.12);
}
.health-banner-danger {
    background: linear-gradient(135deg, #fff1f2, #fde8e8);
    border: 1.5px solid #dc2626;
    box-shadow: 0 4px 20px rgba(220, 38, 38, 0.12);
}
.health-banner-nocorn {
    background: linear-gradient(135deg, #f8fafc, #f1f5f9);
    border: 1.5px solid #64748b;
    box-shadow: 0 4px 20px rgba(100, 116, 139, 0.1);
}
.health-icon { font-size: 40px; flex-shrink: 0; }
.health-title { font-size: 18px; font-weight: 700; margin: 0; }
.health-desc  { font-size: 13px; margin: 3px 0 0 0; opacity: 0.8; }

/* ── Glass KPI cards ── */
.kpi-card {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 22px 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06), 0 8px 32px rgba(0,0,0,0.04);
    border: 1px solid rgba(255,255,255,0.6);
    border-top: 4px solid #2B6CB0;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    animation: fadeUp 0.5s ease both;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.10), 0 16px 48px rgba(0,0,0,0.06);
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.kpi-label {
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 32px;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.1;
    margin-bottom: 6px;
}
.kpi-sub { font-size: 12px; color: #94A3B8; }

/* ── Grade badge ── */
.grade-badge {
    display: inline-block;
    padding: 3px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    margin-top: 6px;
}
.grade-A { background:#dcfce7; color:#166534; }
.grade-B { background:#dbeafe; color:#1e40af; }
.grade-C { background:#fef9c3; color:#854d0e; }
.grade-D { background:#fee2e2; color:#991b1b; }

/* ── Section header ── */
.section-hdr {
    font-size: 15px;
    font-weight: 700;
    color: #1e293b;
    margin: 0 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #f1f5f9;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Variety chip ── */
.variety-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #f0fdf4;
    border: 1.5px solid #16a34a;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 13px;
    color: #166534;
    font-weight: 500;
    margin-bottom: 16px;
    animation: fadeUp 0.4s ease;
}

/* ── Progress bars ── */
.pbar-wrap { margin-bottom: 14px; }
.pbar-meta {
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
    font-size: 13px;
    font-weight: 500;
    color: #334155;
}
.pbar-track {
    background: #f1f5f9;
    border-radius: 999px;
    height: 9px;
    overflow: hidden;
}
.pbar-fill {
    height: 9px;
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* ── Storage card ── */
.st-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px;
}
.st-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 13px;
}
.st-key { color: #64748b; font-weight: 500; }
.st-val { color: #0f172a; font-weight: 600; }

/* ── Recommendation box ── */
.rec-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 12px;
    padding: 16px;
    font-size: 13px;
    color: #92400e;
    line-height: 1.65;
}

/* ── Legend card ── */
.legend-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 22px;
    height: 100%;
}
.legend-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}
.legend-dot {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    flex-shrink: 0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 72px 40px;
    background: linear-gradient(135deg, #f8fafc, #f1f5f9);
    border-radius: 20px;
    border: 2px dashed #cbd5e1;
    margin-top: 20px;
    animation: fadeUp 0.6s ease;
}

/* ── Primary Action Button (Run Quality Assessment) ── */
button[kind="primary"] {
    background: linear-gradient(135deg, #15803d 0%, #16a34a 50%, #059669 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px !important;
    padding: 14px 28px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 14px rgba(22, 163, 74, 0.35) !important;
    transition: all 0.2s ease !important;
    margin-top: 10px !important;
    margin-bottom: 15px !important;
}
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #166534 0%, #15803d 50%, #047857 100%) !important;
    box-shadow: 0 6px 20px rgba(22, 163, 74, 0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Floating AgrowGrow Chatbot Button (Fixed at Bottom-Right Corner) ── */
div[data-testid="stPopover"] {
    position: fixed !important;
    bottom: 28px !important;
    right: 28px !important;
    left: auto !important;
    width: auto !important;
    max-width: fit-content !important;
    z-index: 999999 !important;
}

div[data-testid="stPopover"] > button {
    background: linear-gradient(135deg, #16a34a 0%, #059669 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border-radius: 999px !important;
    padding: 13px 24px !important;
    border: 2px solid rgba(255,255,255,0.6) !important;
    box-shadow: 0 6px 24px rgba(22, 163, 74, 0.5) !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
}

div[data-testid="stPopover"] > button:hover {
    transform: translateY(-3px) scale(1.04) !important;
    box-shadow: 0 10px 32px rgba(22, 163, 74, 0.65) !important;
    background: linear-gradient(135deg, #15803d 0%, #047857 100%) !important;
}

/* Floating popover dialog window */
div[data-testid="stPopoverBody"] {
    max-width: 480px !important;
    min-width: 360px !important;
    max-height: 80vh !important;
    border-radius: 18px !important;
    box-shadow: 0 16px 48px rgba(0,0,0,0.22) !important;
    border: 1.5px solid #cbd5e1 !important;
    padding: 16px !important;
    background: #ffffff !important;
    overflow-y: auto !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
for key, default in {
    "history":          [],
    "current_analysis": None,
    "chat_open":        False,
    "chat_messages":    [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Ensure chat_context is always defined globally to prevent NameError
chat_context = None
use_device = None

# ── Init chatbot (not cached — re-reads GEMINI_API_KEY from .env every restart) ─────
if "_chatbot" not in st.session_state:
    st.session_state["_chatbot"] = CornChatbot()
chatbot = st.session_state["_chatbot"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌽 AgroGrow")
    st.markdown("---")

    st.markdown("### 🌡️ Storage Parameters")
    storage_type = st.selectbox("Storage Type", global_config.storage_types, index=0)

    if storage_type == "Cold Storage":
        temp_input = st.slider("Temperature (°C)", -20.0, 10.0, -2.0, 0.5,
                               help="Sub-zero to chilled range for cold storage.")
        humidity_input = st.slider("Humidity (%)", 30.0, 90.0, 50.0, 1.0)
    elif storage_type == "Hermetic Bag":
        temp_input = st.slider("Temperature (°C)", 5.0, 40.0, 22.0, 0.5)
        humidity_input = st.slider("Humidity (%)", 30.0, 95.0, 65.0, 1.0)
    else:
        temp_input = st.slider("Temperature (°C)", 10.0, 50.0, 28.0, 0.5)
        humidity_input = st.slider("Humidity (%)", 30.0, 98.0, 75.0, 1.0)

    st.markdown("---")
    st.markdown("### 🌾 Grain Variety")
    corn_variety_input = st.selectbox(
        "Corn Variety Mode",
        options=[
            "🔍 Auto-Detect Variety",
            "🌽 Dent Corn (Yellow/White) — Commercial",
            "🎨 Indian / Flint Corn (Multicoloured)",
            "🍬 Sweet Corn (Pale Cream/Yellow)",
            "🍿 Popcorn (Small Hard Kernels)",
            "🔵 Blue / Black Corn (Hopi variety)",
        ],
        index=0,
        help="Select manually for unusual varieties to prevent misclassification of pigmentation as disease."
    )



# ── Animated Header ───────────────────────────────────────────────────────────
st.markdown("""
<div class='ag-header'>
  <p class='ag-title'>🌽 AgroGrow</p>
  <p class='ag-subtitle'>Post-Harvest Corn Quality Assessment &amp; Shelf-Life Forecasting</p>
  <span class='ag-pill'>Semantic Segmentation · AI Grading · Storage Prediction</span>
</div>
""", unsafe_allow_html=True)

# ── File Upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📁 Upload Corn Cob Image (JPG, JPEG, PNG, WEBP, BMP, etc.)",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"],
    help="Upload a photograph of a corn cob in any format. The system automatically detects and analyzes the ear."
)

if uploaded_file is not None:
    # Invalidate cache on new image
    if st.session_state.get("last_uploaded_name") != uploaded_file.name:
        st.session_state.last_uploaded_name = uploaded_file.name
        st.session_state.current_analysis = None

    # Robust in-memory decoding supporting any format (JPG, PNG with/without alpha, WEBP, BMP)
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    _raw_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if _raw_bgr is None:
        st.error("❌ Unable to decode the uploaded image file. Please verify it is a valid image.")
        st.stop()

    _raw_rgb = cv2.cvtColor(_raw_bgr, cv2.COLOR_BGR2RGB)
    _H, _W = _raw_rgb.shape[:2]

    # Save temporary file with appropriate image extension
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        ext = ".jpg"
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    cv2.imwrite(tfile.name, _raw_bgr)
    tfile.close()
    img_path = Path(tfile.name)

    from AgroGrow.prediction.predictor import auto_crop_corn_ear
    _auto_cropped, _auto_bbox = auto_crop_corn_ear(_raw_rgb)
    _ax, _ay, _aw, _ah = _auto_bbox
    _is_full = (_aw >= _W * 0.95 and _ah >= _H * 0.95)

    # ── Framing mode ─────────────────────────────────────────────────────────
    st.markdown("<p style='font-size:14px;font-weight:700;color:#1e293b;margin:14px 0 6px 0;'>✂️ Cob Framing Mode</p>", unsafe_allow_html=True)
    crop_mode = st.segmented_control(
        "Cob Framing Mode",
        options=["🌽 Auto Crop Ear", "📷 Full Image"],
        default=st.session_state.get("last_crop_mode", "🌽 Auto Crop Ear"),
        label_visibility="collapsed"
    )
    if not crop_mode:
        crop_mode = "🌽 Auto Crop Ear"

    if st.session_state.get("last_crop_mode") != crop_mode:
        st.session_state.last_crop_mode = crop_mode
        st.session_state.current_analysis = None

    # ── Determine inference image ─────────────────────────────────────────────
    if crop_mode == "🌽 Auto Crop Ear":
        if _is_full and _W > _H * 1.1:
            _target_crop = _raw_rgb[:, int(_W * 0.10):int(_W * 0.50)]
        else:
            _target_crop = _auto_cropped
        _c = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        cv2.imwrite(_c.name, cv2.cvtColor(_target_crop, cv2.COLOR_RGB2BGR))
        _c.close()
        inference_img_path = Path(_c.name)
    else:
        _target_crop = _raw_rgb
        inference_img_path = img_path

    # ── Image Preview ─────────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("<p class='section-hdr'>📷 Original Image</p>", unsafe_allow_html=True)
        st.image(_raw_rgb, use_container_width=True, caption=f"Original — {_W}×{_H} px")
    with v2:
        st.markdown("<p class='section-hdr'>🎯 Region for Analysis</p>", unsafe_allow_html=True)
        st.image(_target_crop, use_container_width=True,
                 caption=f"Model Input — {_target_crop.shape[1]}×{_target_crop.shape[0]} px")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Run Analysis Button (Immediately below the images) ────────────────────
    if st.button("🔍  Run Quality Assessment Pipeline", use_container_width=True, type="primary", key="btn_run_assessment"):
        with st.spinner("Analysing kernels — please wait..."):
            try:
                predictor = CornPredictor(device=use_device)
                mask, overlay, _, confidence = predictor.predict_single(
                    inference_img_path, auto_crop=False, corn_variety=corn_variety_input
                )
                global_config.results_dir.mkdir(parents=True, exist_ok=True)
                overlay_temp_path = global_config.results_dir / f"{img_path.stem}_overlay.png"
                cv2.imwrite(str(overlay_temp_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                features = CornFeatureExtractor.extract_features(mask)
                grading  = CornGrader.classify_grade(features, confidence)

                try:
                    sp = StoragePredictor()
                    shelf_life, risk_level, recommendation = sp.predict(
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
                    logger.warning(f"Storage model failed: {e}")
                    shelf_life = 45.0
                    risk_level = "Medium Risk"
                    recommendation = "Aerate grain and store in a cool, low-moisture silo."

                st.session_state.current_analysis = {
                    "image_path":   img_path,
                    "overlay_path": overlay_temp_path,
                    "features":     features,
                    "grading":      grading,
                    "variety":      getattr(predictor, "last_detected_variety",
                                           "Commercial Dent Corn (Zea mays indentata)"),
                    "storage": {
                        "shelf_life_days": shelf_life,
                        "risk_level":      risk_level,
                        "recommendation":  recommendation,
                        "temperature_c":   temp_input,
                        "humidity_pct":    humidity_input,
                        "storage_type":    storage_type
                    }
                }

                st.session_state.history.append({
                    "Filename":      uploaded_file.name,
                    "Grade":         grading["grade"],
                    "Quality Score": f"{features['quality_score']:.1f}/100",
                    "Shelf Life":    f"{shelf_life:.0f} Days",
                    "Risk":          risk_level
                })

                # Reset chat on new analysis
                st.session_state.chat_messages = []

            except Exception as ex:
                st.error(f"❌ Assessment Error: {ex}")
                logger.exception(ex)

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.current_analysis is not None:
        res = st.session_state.current_analysis
        f   = res["features"]
        g   = res["grading"]
        s   = res["storage"]
        variety_name = res.get("variety", "Commercial Dent Corn (Zea mays indentata)")

        # Build chatbot context from current results
        chat_context = {
            **f,
            **g,
            **s,
            "variety": variety_name
        }

        st.divider()

        # Variety chip
        var_emoji = "🌽"
        if "flint" in variety_name.lower() or "indian" in variety_name.lower():
            var_emoji = "🎨"
        elif "sweet" in variety_name.lower():
            var_emoji = "🍬"
        elif "popcorn" in variety_name.lower() or "everta" in variety_name.lower():
            var_emoji = "🍿"
        elif "blue" in variety_name.lower() or "black" in variety_name.lower():
            var_emoji = "🔵"

        st.markdown(f"""
        <div class='variety-chip' style='margin-bottom:20px;'>
            <span style='font-size:22px;'>{var_emoji}</span>
            <span><strong>Detected Variety:</strong> {variety_name}</span>
        </div>""", unsafe_allow_html=True)

        # ── Health Status Banner ────────────────────────────────────────────────
        no_corn = f.get("no_corn_detected", False)
        q = f["quality_score"]
        d = f["disease_percentage"]

        if no_corn:
            st.markdown("""
            <div style='background:#fef2f2;border:2px solid #ef4444;border-radius:12px;padding:16px 20px;margin-bottom:20px;display:flex;align-items:center;gap:14px;'>
                <div style='font-size:32px;'>🔍</div>
                <div>
                    <h3 style='margin:0;color:#991b1b;font-size:18px;'>No Corn Detected in Image</h3>
                    <p style='margin:4px 0 0 0;color:#b91c1c;font-size:13px;'>The uploaded image does not appear to contain a valid corn cob or ear. Please upload a clear photo of corn.</p>
                </div>
            </div>""", unsafe_allow_html=True)
        elif q >= 75 and d < 5.0:
            st.markdown(f"""
            <div style='background:#f0fdf4;border:2px solid #22c55e;border-radius:12px;padding:16px 20px;margin-bottom:20px;display:flex;align-items:center;gap:14px;'>
                <div style='font-size:32px;'>✅</div>
                <div>
                    <h3 style='margin:0;color:#166534;font-size:18px;'>Corn Batch is Healthy</h3>
                    <p style='margin:4px 0 0 0;color:#15803d;font-size:13px;'>Overall quality score is <strong>{q:.1f}/100</strong> with minimal disease ({d:.1f}%). Safe for standard or extended storage.</p>
                </div>
            </div>""", unsafe_allow_html=True)
        elif q >= 45 and d < 20.0:
            st.markdown(f"""
            <div style='background:#fffbeb;border:2px solid #f59e0b;border-radius:12px;padding:16px 20px;margin-bottom:20px;display:flex;align-items:center;gap:14px;'>
                <div style='font-size:32px;'>⚠️</div>
                <div>
                    <h3 style='margin:0;color:#92400e;font-size:18px;'>Moderate Defects Detected</h3>
                    <p style='margin:4px 0 0 0;color:#b45309;font-size:13px;'>Quality score is <strong>{q:.1f}/100</strong> with {d:.1f}% diseased/damaged kernels. Immediate aeration or drying recommended.</p>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#fef2f2;border:2px solid #ef4444;border-radius:12px;padding:16px 20px;margin-bottom:20px;display:flex;align-items:center;gap:14px;'>
                <div style='font-size:32px;'>❌</div>
                <div>
                    <h3 style='margin:0;color:#991b1b;font-size:18px;'>Severe Disease / Rotten Corn Detected</h3>
                    <p style='margin:4px 0 0 0;color:#b91c1c;font-size:13px;'>Quality score is critical at <strong>{q:.1f}/100</strong> with {d:.1f}% diseased kernels. Not recommended for long-term storage or commercial sale.</p>
                </div>
            </div>""", unsafe_allow_html=True)

        grade_text  = g["grade"]
        raw_conf    = g["confidence"]
        conf_pct    = raw_conf * 100.0 if raw_conf <= 1.0 else raw_conf
        grade_ltr   = grade_text.replace("Grade ", "") if "Grade" in grade_text else "D"
        risk_col    = ("#16a34a" if "Low" in s["risk_level"] else
                       ("#d97706" if "Medium" in s["risk_level"] else "#dc2626"))

        # ── Section 1: Overlay + Storage Analysis Side-by-Side ──────────────────
        res_col_left, res_col_right = st.columns([1.2, 1], gap="large")

        with res_col_left:
            st.markdown("<p class='section-hdr'>🎨 Segmentation Overlay</p>", unsafe_allow_html=True)
            st.image(str(res["overlay_path"]), use_container_width=True,
                     caption="CornNet semantic segmentation — natural photo blend")

            # Small, basic, clean inline color legend
            st.markdown("""
            <div style='display:flex;gap:16px;justify-content:center;align-items:center;padding:9px 14px;background:#f8fafc;border-radius:10px;border:1px solid #e2e8f0;margin-top:8px;font-size:12px;font-weight:600;color:#334155;'>
                <span style='display:flex;align-items:center;gap:6px;'><span style='width:10px;height:10px;border-radius:50%;background:#00CC00;display:inline-block;'></span> Healthy</span>
                <span style='display:flex;align-items:center;gap:6px;'><span style='width:10px;height:10px;border-radius:50%;background:#0066FF;display:inline-block;'></span> Missing</span>
                <span style='display:flex;align-items:center;gap:6px;'><span style='width:10px;height:10px;border-radius:50%;background:#FF2222;display:inline-block;'></span> Diseased</span>
            </div>
            """, unsafe_allow_html=True)

        with res_col_right:
            st.markdown("<p class='section-hdr'>🏭 Storage Analysis</p>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='st-card'>
                <div class='st-row'><span class='st-key'>Storage Mode</span><span class='st-val'>{s["storage_type"]}</span></div>
                <div class='st-row'><span class='st-key'>Temperature</span><span class='st-val'>{s["temperature_c"]} °C</span></div>
                <div class='st-row'><span class='st-key'>Humidity</span><span class='st-val'>{s["humidity_pct"]} % RH</span></div>
                <div class='st-row'><span class='st-key'>Predicted Shelf Life</span>
                    <span class='st-val' style='color:{risk_col};'>{s["shelf_life_days"]:.0f} days</span></div>
                <div class='st-row'><span class='st-key'>Risk Category</span>
                    <span class='st-val' style='color:{risk_col};'>{s["risk_level"]}</span></div>
                <div class='st-row' style='border-bottom:none;'>
                    <span class='st-key'>Quality Grade</span><span class='st-val'>{grade_text}</span></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.divider()

        # ── Section 2: Quality Metrics (KPI Cards as before) ───────────────────
        st.markdown("<p class='section-hdr'>📊 Quality Metrics</p>", unsafe_allow_html=True)

        kc1, kc2, kc3, kc4 = st.columns(4)
        with kc1:
            st.markdown(f"""
            <div class='kpi-card' style='border-top-color:#3b82f6;animation-delay:0s;'>
                <div class='kpi-label'>Quality Grade</div>
                <div class='kpi-value'>{grade_text}</div>
                <span class='grade-badge grade-{grade_ltr}'>{conf_pct:.1f}% conf.</span>
            </div>""", unsafe_allow_html=True)
        with kc2:
            st.markdown(f"""
            <div class='kpi-card' style='border-top-color:#22c55e;animation-delay:0.07s;'>
                <div class='kpi-label'>Quality Score</div>
                <div class='kpi-value'>{f["quality_score"]:.1f}<span style='font-size:16px;font-weight:400;color:#94a3b8;'>/100</span></div>
                <div class='kpi-sub'>Healthy: {f["healthy_percentage"]:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with kc3:
            st.markdown(f"""
            <div class='kpi-card' style='border-top-color:#f59e0b;animation-delay:0.14s;'>
                <div class='kpi-label'>Estimated Shelf Life</div>
                <div class='kpi-value'>{s["shelf_life_days"]:.0f}<span style='font-size:16px;font-weight:400;color:#94a3b8;'> days</span></div>
                <div class='kpi-sub'>{s["temperature_c"]}°C · {s["humidity_pct"]}% RH</div>
            </div>""", unsafe_allow_html=True)
        with kc4:
            st.markdown(f"""
            <div class='kpi-card' style='border-top-color:{risk_col};animation-delay:0.21s;'>
                <div class='kpi-label'>Storage Risk</div>
                <div class='kpi-value' style='color:{risk_col};font-size:22px;'>{s["risk_level"]}</div>
                <div class='kpi-sub'>{s["storage_type"]}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.divider()

        # ── Section 3: Grade Summary + Recommendation (as before) ─────────────
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("<p class='section-hdr'>📋 Grade Summary</p>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='padding:16px;background:#f8fafc;border-radius:12px;border:1px solid #e2e8f0;
                        font-size:13px;color:#334155;line-height:1.65;'>
                {g["summary"]}
            </div>""", unsafe_allow_html=True)
        with sc2:
            st.markdown("<p class='section-hdr'>💡 Storage Recommendation</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='rec-box'>{s['recommendation']}</div>", unsafe_allow_html=True)

        st.divider()

        # ── Section 4: PDF Report ──────────────────────────────────────────────
        st.markdown("<p class='section-hdr'>📄 Quality Assessment Report</p>", unsafe_allow_html=True)
        pc1, pc2 = st.columns([2, 1])
        with pc1:
            st.markdown("""
            <div style='font-size:13px;color:#64748b;line-height:1.65;padding:8px 0;'>
                Download a formatted PDF certificate containing the segmentation overlay,
                storage projections, and official classification grade for this batch.
            </div>""", unsafe_allow_html=True)
        with pc2:
            report_name = f"{img_path.stem}_quality_report.pdf"
            pdf_path    = global_config.reports_dir / report_name
            if st.button("🛠️ Compile PDF Report", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    try:
                        PDFReportGenerator.generate(
                            image_path=res["image_path"],
                            overlay_path=res["overlay_path"],
                            features=f, grading=g, storage=s,
                            output_pdf_path=pdf_path
                        )
                        st.success(f"✅ {report_name}")
                    except Exception as e:
                        st.error(f"PDF failed: {e}")
                        logger.exception(e)
            if pdf_path.exists():
                with open(pdf_path, "rb") as pf:
                    st.download_button("📥 Download PDF", data=pf,
                                       file_name=report_name, mime="application/pdf",
                                       use_container_width=True)

        st.divider()

        # ── Section 5: History ────────────────────────────────────────────────
        st.markdown("<p class='section-hdr'>🕒 Assessment History</p>", unsafe_allow_html=True)
        if st.session_state.history:
            st.dataframe(pd.DataFrame(st.session_state.history),
                         use_container_width=True, hide_index=True)
        else:
            st.caption("No assessments run yet.")

else:
    # ── Disabled Run Button for Initial Clarity ──────────────────────────────
    st.button(
        "🔍  Run Quality Assessment Pipeline",
        use_container_width=True,
        disabled=True,
        help="Please upload a corn cob image above to enable assessment",
        key="btn_run_assessment_disabled"
    )
    st.caption("👆 Upload an image above (.jpg, .jpeg, .png, etc.) to activate quality assessment.")

    # ── Empty State ───────────────────────────────────────────────────────────
    chat_context = None
    st.markdown("""
    <div class='empty-state'>
        <div style='font-size:64px;margin-bottom:16px;'>🌽</div>
        <h3 style='color:#1e293b;font-weight:700;margin:0 0 10px 0;font-size:22px;'>
            Upload a Corn Image to Get Started
        </h3>
        <p style='color:#64748b;font-size:14px;max-width:520px;margin:0 auto;line-height:1.7;'>
            Upload a photo of a corn cob in any standard image format (.jpg, .jpeg, .png, .webp, .bmp).
            The system will automatically detect the ear, run semantic segmentation,
            classify the grain variety, evaluate USDA/ISO grade, and forecast shelf life.
        </p>
        <div style='display:flex;gap:16px;justify-content:center;margin-top:24px;flex-wrap:wrap;'>
            <div style='background:white;border:1px solid #e2e8f0;border-radius:12px;padding:14px 20px;
                        font-size:13px;color:#475569;'>🔬 Semantic Segmentation</div>
            <div style='background:white;border:1px solid #e2e8f0;border-radius:12px;padding:14px 20px;
                        font-size:13px;color:#475569;'>📊 USDA / ISO Grading</div>
            <div style='background:white;border:1px solid #e2e8f0;border-radius:12px;padding:14px 20px;
                        font-size:13px;color:#475569;'>📦 Shelf-Life Forecast</div>
            <div style='background:white;border:1px solid #e2e8f0;border-radius:12px;padding:14px 20px;
                        font-size:13px;color:#475569;'>🌾 5 Variety Detection</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    chat_context = None

# ── Floating AgroGrow Chatbot Popover (Always anchored at Bottom-Right) ──────────
bot_label = "💬 Ask AgroGrow AI" 
with st.popover(bot_label, help="Chat with AgroGrow Virtual Agronomist"):
    st.markdown("#### 🌽 AgroGrow AI Assistant")
    st.caption("Ask me anything about corn quality, storage, or diseases.")
    st.divider()

    # Scrollable chat messages container
    chat_container = st.container(height=300)
    with chat_container:
        if not st.session_state.chat_messages:
            st.caption("Type your question below to get started.")
        else:
            for msg in st.session_state.chat_messages:
                avatar = "🤖" if msg["role"] == "assistant" else "👤"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])

    # Chat Input
    user_q = st.chat_input("Ask about corn quality, varieties, storage...", key="AgrowGrow_popover_input")
    if user_q:
        st.session_state.chat_messages.append({"role": "user", "content": user_q})
        with st.spinner("Thinking..."):
            reply = chatbot.chat(user_q, analysis_context=chat_context)
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()

    # Clear chat button
    if st.session_state.chat_messages:
        if st.button("🗑️ Clear Chat", key="btn_clear_chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
