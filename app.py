"""
AgroGrow Interactive Streamlit Dashboard.
Provides a clean web interface for upload, segmentation mapping,
agronomic quality analytics, and shelf-life model predictions.
"""

import tempfile
import sys
from pathlib import Path
import streamlit as st
import numpy as np
import cv2
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
from AgroGrow.utils.report_generator import PDFReportGenerator

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroGrow — Corn Quality Assessment",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* ── Metric KPI cards ── */
        .kpi-card {
            background: #ffffff;
            border-radius: 14px;
            padding: 22px 20px 18px 20px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
            border-top: 4px solid #2B6CB0;
            text-align: center;
            margin-bottom: 4px;
        }
        .kpi-label {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: #94A3B8;
            margin-bottom: 6px;
        }
        .kpi-value {
            font-size: 30px;
            font-weight: 700;
            color: #0F172A;
            line-height: 1.1;
            margin-bottom: 6px;
        }
        .kpi-sub {
            font-size: 12px;
            color: #94A3B8;
            font-weight: 400;
        }

        /* ── Grade badge ── */
        .grade-badge {
            display: inline-block;
            padding: 3px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            margin-top: 6px;
        }
        .grade-A { background: #DCFCE7; color: #166534; }
        .grade-B { background: #DBEAFE; color: #1E40AF; }
        .grade-C { background: #FEF9C3; color: #854D0E; }
        .grade-D { background: #FEE2E2; color: #991B1B; }

        /* ── Section header ── */
        .section-header {
            font-size: 16px;
            font-weight: 600;
            color: #1E293B;
            margin: 0 0 12px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #F1F5F9;
        }

        /* ── Colour legend chips ── */
        .legend-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        .legend-dot {
            width: 14px; height: 14px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .legend-label { font-size: 13px; color: #334155; font-weight: 500; }

        /* ── Info card (variety banner) ── */
        .variety-banner {
            background: #F0FDF4;
            border-left: 4px solid #16A34A;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 16px;
            font-size: 13px;
            color: #166534;
        }

        /* ── Storage info card ── */
        .storage-card {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 20px;
            height: 100%;
        }
        .storage-row {
            display: flex;
            justify-content: space-between;
            padding: 7px 0;
            border-bottom: 1px solid #F1F5F9;
            font-size: 13px;
        }
        .storage-key { color: #64748B; font-weight: 500; }
        .storage-val { color: #0F172A; font-weight: 600; }

        /* ── Recommendation box ── */
        .rec-box {
            background: #FFFBEB;
            border: 1px solid #FDE68A;
            border-radius: 10px;
            padding: 14px 16px;
            font-size: 13px;
            color: #92400E;
            line-height: 1.6;
        }

        /* ── History table ── */
        .history-header {
            font-size: 14px;
            font-weight: 600;
            color: #1E293B;
            margin-bottom: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌽 AgroGrow")
    st.markdown("---")

    # Storage Parameters
    st.markdown("### 🌡️ Storage Parameters")
    storage_type = st.selectbox("Storage Type", global_config.storage_types, index=0)

    if storage_type == "Cold Storage":
        temp_input = st.slider(
            "Temperature (°C)",
            min_value=-20.0, max_value=10.0, value=-2.0, step=0.5,
            help="Cold storage: sub-zero and chilled ranges (-20°C to +10°C)."
        )
        humidity_input = st.slider("Humidity (%)", min_value=30.0, max_value=90.0, value=50.0, step=1.0)
    elif storage_type == "Hermetic Bag":
        temp_input = st.slider(
            "Temperature (°C)",
            min_value=5.0, max_value=40.0, value=22.0, step=0.5,
            help="Hermetically sealed bag, controlled ambient conditions."
        )
        humidity_input = st.slider("Humidity (%)", min_value=30.0, max_value=95.0, value=65.0, step=1.0)
    else:  # Open Air
        temp_input = st.slider(
            "Temperature (°C)",
            min_value=10.0, max_value=50.0, value=28.0, step=0.5,
            help="Open air field / warehouse ambient conditions."
        )
        humidity_input = st.slider("Humidity (%)", min_value=30.0, max_value=98.0, value=75.0, step=1.0)

    st.markdown("---")

    # Grain Variety
    st.markdown("### 🌾 Grain Variety")
    corn_variety_input = st.selectbox(
        "Corn Variety Mode",
        options=[
            "🔍 Auto-Detect Variety",
            "🌾 Indian / Flint / Multi-colored Corn",
            "🌽 Commercial Dent Corn (Yellow/White)"
        ],
        index=0,
        help="For ruby-red or purple pigmented flint corn, select manually to prevent misclassification."
    )

    st.markdown("---")

    # Model Config
    st.markdown("### ⚙️ Model Configuration")
    has_cuda = torch.cuda.is_available()
    device_options = ["Auto (GPU if available)", "CPU (Safe Mode)"] if has_cuda else ["CPU (Safe Mode)"]
    device_choice = st.selectbox("Inference Hardware", device_options, index=0)
    use_device = "cpu" if "CPU" in device_choice else None

    mode_label = "🟢 GPU Accelerated" if (has_cuda and use_device != "cpu") else "🔵 CPU Mode"
    st.info(f"**Active Mode:** {mode_label}")

    weights_file = global_config.weights_dir / "best_model.pth"
    if weights_file.exists():
        st.success("✅ Model Weights Loaded")
    else:
        st.warning("⚠️ No weights found — train first")

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 8px 0 4px 0;'>
  <h1 style='font-size:32px; font-weight:700; color:#0F172A; margin:0;'>🌽 AgroGrow</h1>
  <p style='color:#64748B; font-size:15px; margin:4px 0 0 0;'>
      Corn Quality Assessment &amp; Post-Harvest Shelf-Life Forecasting
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.divider()

# ── File Upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📁 Upload a corn image (.jpg / .jpeg)",
    type=["jpg", "jpeg"],
    help="Upload a close-up photograph of a corn cob. The system will auto-detect and focus on the ear."
)

if uploaded_file is not None:
    # Invalidate stale cache on new image upload
    if st.session_state.get("last_uploaded_name") != uploaded_file.name:
        st.session_state.last_uploaded_name = uploaded_file.name
        st.session_state.current_analysis = None

    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tfile.write(uploaded_file.getvalue())
    tfile.close()
    img_path = Path(tfile.name)

    # Load and auto-crop
    from AgroGrow.prediction.predictor import auto_crop_corn_ear
    _raw_bgr = cv2.imread(str(img_path))
    _raw_rgb = cv2.cvtColor(_raw_bgr, cv2.COLOR_BGR2RGB)
    _H, _W = _raw_rgb.shape[:2]
    _auto_cropped, _auto_bbox = auto_crop_corn_ear(_raw_rgb)
    _ax, _ay, _aw, _ah = _auto_bbox
    _is_full = (_aw >= _W * 0.95 and _ah >= _H * 0.95)

    # ── Framing Control Bar ───────────────────────────────────────────────────
    st.markdown("#### ✂️ Cob Framing Mode")
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        crop_mode = st.radio(
            "Select framing:",
            options=["🌽 Auto Crop Ear", "🖐 Manual Crop", "📷 Full Image"],
            index=0,
            horizontal=True,
            label_visibility="collapsed"
        )
    if st.session_state.get("last_crop_mode") != crop_mode:
        st.session_state.last_crop_mode = crop_mode
        st.session_state.current_analysis = None

    with f_col2:
        if crop_mode == "🌽 Auto Crop Ear":
            if _is_full:
                st.info("ℹ️ Full image — no crop needed.")
            else:
                st.success(f"✅ Ear auto-detected ({_aw}×{_ah} px) — background suppressed.")
        elif crop_mode == "🖐 Manual Crop":
            st.info("💡 Use sliders below to frame the corn ear.")
        else:
            st.info("📷 Full photograph analyzed with background suppression.")

    # ── Determine inference image ─────────────────────────────────────────────
    if crop_mode == "🌽 Auto Crop Ear":
        if _is_full and _W > _H * 1.1:
            _target_crop = _raw_rgb[:, int(_W * 0.10):int(_W * 0.50)]
        else:
            _target_crop = _auto_cropped
        _c = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(_c.name, cv2.cvtColor(_target_crop, cv2.COLOR_RGB2BGR))
        _c.close()
        inference_img_path = Path(_c.name)

    elif crop_mode == "🖐 Manual Crop":
        with st.expander("🛠️ Manual Framing Controls", expanded=True):
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

            s1, s2 = st.columns(2)
            with s1:
                cx1 = st.slider("Left Boundary", 0, _W - 10, init_x1, key="slider_x1")
                cx2 = st.slider("Right Boundary", cx1 + 10, _W, max(cx1 + 10, init_x2), key="slider_x2")
            with s2:
                cy1 = st.slider("Top Boundary", 0, _H - 10, init_y1, key="slider_y1")
                cy2 = st.slider("Bottom Boundary", cy1 + 10, _H, max(cy1 + 10, init_y2), key="slider_y2")

        _target_crop = _raw_rgb[cy1:cy2, cx1:cx2]
        _c = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(_c.name, cv2.cvtColor(_target_crop, cv2.COLOR_RGB2BGR))
        _c.close()
        inference_img_path = Path(_c.name)

    else:  # Full Image
        _target_crop = _raw_rgb
        inference_img_path = img_path

    # ── Preview Images ────────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.markdown("<p class='section-header'>📷 Original Image</p>", unsafe_allow_html=True)
        st.image(_raw_rgb, use_container_width=True, caption=f"Original — {_W}×{_H} px")
    with v_col2:
        st.markdown("<p class='section-header'>🎯 Region for Analysis</p>", unsafe_allow_html=True)
        st.image(_target_crop, use_container_width=True,
                 caption=f"Model Input — {_target_crop.shape[1]}×{_target_crop.shape[0]} px")

    # ── Run Analysis Button ───────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("🔍  Run Quality Assessment", use_container_width=True, type="primary"):
        with st.spinner("Analysing kernels — please wait..."):
            try:
                # 1. Segmentation
                predictor = CornPredictor(device=use_device)
                mask, overlay, _, confidence = predictor.predict_single(
                    inference_img_path, auto_crop=False, corn_variety=corn_variety_input
                )

                # Save overlay
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
                    logger.warning(f"Storage model load failed: {e}. Using heuristic fallback.")
                    shelf_life = 45.0
                    risk_level = "Medium Risk"
                    recommendation = "Aerate grain and store in a cool, low-moisture silo."

                # Cache results
                st.session_state.current_analysis = {
                    "image_path": img_path,
                    "overlay_path": overlay_temp_path,
                    "features": features,
                    "grading": grading,
                    "variety": getattr(predictor, "last_detected_variety", "Commercial Dent Corn (Yellow/White)"),
                    "storage": {
                        "shelf_life_days": shelf_life,
                        "risk_level": risk_level,
                        "recommendation": recommendation,
                        "temperature_c": temp_input,
                        "humidity_pct": humidity_input,
                        "storage_type": storage_type
                    }
                }

                # History log
                st.session_state.history.append({
                    "Filename": uploaded_file.name,
                    "Grade": grading["grade"],
                    "Quality Score": f"{features['quality_score']:.1f}/100",
                    "Shelf Life": f"{shelf_life:.1f} Days",
                    "Risk": risk_level
                })

            except Exception as ex:
                st.error(f"❌ Assessment Error: {ex}")
                logger.exception(ex)

    # ── Results Display ───────────────────────────────────────────────────────
    if st.session_state.current_analysis is not None:
        res = st.session_state.current_analysis
        f = res["features"]
        g = res["grading"]
        s = res["storage"]

        st.divider()

        # Variety banner (only for flint corn)
        variety_name = res.get("variety", "")
        if "flint" in variety_name.lower() or "indian" in variety_name.lower():
            st.markdown(f"""
            <div class='variety-banner'>
                🌾 <b>Identified Variety:</b> {variety_name}<br>
                <span style='font-size:12px;'>
                Ruby-red and purple anthocyanin pigmentation recognised as healthy grain — not disease.
                </span>
            </div>""", unsafe_allow_html=True)

        # ── Section 1: Segmentation Overlay + Legend ──────────────────────────
        st.markdown("<p class='section-header'>🎨 Segmentation Overlay</p>", unsafe_allow_html=True)
        ov_c1, ov_c2 = st.columns([1.4, 1])

        with ov_c1:
            st.image(str(res["overlay_path"]), use_container_width=True,
                     caption="CornNet segmentation — natural photo blend")

        with ov_c2:
            st.markdown("""
            <div style='padding:20px; background:#F8FAFC; border-radius:12px; border:1px solid #E2E8F0; height:100%;'>
                <p style='font-size:13px; font-weight:600; color:#1E293B; margin:0 0 16px 0;'>Colour Legend</p>
                <div class='legend-row'>
                    <div class='legend-dot' style='background:#00CC00;'></div>
                    <div class='legend-label'>Healthy Kernels</div>
                </div>
                <div class='legend-row'>
                    <div class='legend-dot' style='background:#0066FF;'></div>
                    <div class='legend-label'>Missing Kernel Sockets</div>
                </div>
                <div class='legend-row'>
                    <div class='legend-dot' style='background:#FF2222;'></div>
                    <div class='legend-label'>Diseased / Rotten Kernels</div>
                </div>
                <div style='height:1px; background:#E2E8F0; margin:14px 0;'></div>
                <p style='font-size:11px; color:#94A3B8; line-height:1.5; margin:0;'>
                    Overlay blends predicted kernel classes over the original photograph at 55% opacity.
                    Background pixels remain uncoloured.
                </p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Section 2: KPI Cards ──────────────────────────────────────────────
        st.markdown("<p class='section-header'>📊 Quality Metrics</p>", unsafe_allow_html=True)

        grade_text = g["grade"]
        raw_conf = g["confidence"]
        conf_pct = raw_conf * 100.0 if raw_conf <= 1.0 else raw_conf

        grade_letter = grade_text.replace("Grade ", "") if "Grade" in grade_text else "D"
        grade_class = f"grade-{grade_letter}"

        kc1, kc2, kc3, kc4 = st.columns(4)

        risk_color = "#16A34A" if "Low" in s["risk_level"] else ("#D97706" if "Medium" in s["risk_level"] else "#DC2626")

        with kc1:
            st.markdown(f"""
            <div class='kpi-card' style='border-top-color:#3B82F6;'>
                <div class='kpi-label'>Quality Grade</div>
                <div class='kpi-value'>{grade_text}</div>
                <span class='grade-badge {grade_class}'>{conf_pct:.1f}% confidence</span>
            </div>""", unsafe_allow_html=True)

        with kc2:
            st.markdown(f"""
            <div class='kpi-card' style='border-top-color:#22C55E;'>
                <div class='kpi-label'>Quality Score</div>
                <div class='kpi-value'>{f["quality_score"]:.1f}<span style='font-size:16px;font-weight:400;color:#94A3B8;'>/100</span></div>
                <div class='kpi-sub'>Healthy: {f["healthy_percentage"]:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        with kc3:
            st.markdown(f"""
            <div class='kpi-card' style='border-top-color:#F59E0B;'>
                <div class='kpi-label'>Estimated Shelf Life</div>
                <div class='kpi-value'>{s["shelf_life_days"]:.0f}<span style='font-size:16px;font-weight:400;color:#94A3B8;'> days</span></div>
                <div class='kpi-sub'>{s["temperature_c"]}°C · {s["humidity_pct"]}% RH</div>
            </div>""", unsafe_allow_html=True)

        with kc4:
            st.markdown(f"""
            <div class='kpi-card' style='border-top-color:{risk_color};'>
                <div class='kpi-label'>Storage Risk</div>
                <div class='kpi-value' style='color:{risk_color};font-size:22px;'>{s["risk_level"]}</div>
                <div class='kpi-sub'>{s["storage_type"]}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.divider()

        # ── Section 3: Kernel Breakdown + Storage Details ─────────────────────
        det_c1, det_c2 = st.columns(2)

        with det_c1:
            st.markdown("<p class='section-header'>🌱 Kernel Composition</p>", unsafe_allow_html=True)

            healthy_pct = f["healthy_percentage"]
            missing_pct = f["missing_percentage"]
            disease_pct = f["disease_percentage"]

            # Progress-bar style breakdown — clean and minimal
            def pct_bar(label, value, color, icon):
                bar_w = max(0.5, min(value, 100))
                st.markdown(f"""
                <div style='margin-bottom:14px;'>
                    <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                        <span style='font-size:13px; font-weight:500; color:#334155;'>{icon} {label}</span>
                        <span style='font-size:13px; font-weight:600; color:#0F172A;'>{value:.2f}%</span>
                    </div>
                    <div style='background:#F1F5F9; border-radius:999px; height:8px; overflow:hidden;'>
                        <div style='width:{bar_w}%; background:{color}; height:8px; border-radius:999px;'></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            pct_bar("Healthy Kernels", healthy_pct, "#22C55E", "🟢")
            pct_bar("Missing Kernels", missing_pct, "#3B82F6", "🔵")
            pct_bar("Diseased Kernels", disease_pct, "#EF4444", "🔴")

            st.markdown(f"""
            <div style='margin-top:12px; padding:12px; background:#F8FAFC; border-radius:10px; border:1px solid #E2E8F0;'>
                <div style='font-size:12px; color:#64748B; font-weight:500;'>Kernel Density Blobs</div>
                <div style='font-size:18px; font-weight:700; color:#0F172A; margin-top:2px;'>
                    {f["num_healthy_blobs"]} <span style='font-size:13px; font-weight:400; color:#94A3B8;'>healthy regions</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with det_c2:
            st.markdown("<p class='section-header'>🏭 Storage Analysis</p>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class='storage-card'>
                <div class='storage-row'>
                    <span class='storage-key'>Storage Mode</span>
                    <span class='storage-val'>{s["storage_type"]}</span>
                </div>
                <div class='storage-row'>
                    <span class='storage-key'>Temperature</span>
                    <span class='storage-val'>{s["temperature_c"]} °C</span>
                </div>
                <div class='storage-row'>
                    <span class='storage-key'>Humidity</span>
                    <span class='storage-val'>{s["humidity_pct"]} % RH</span>
                </div>
                <div class='storage-row'>
                    <span class='storage-key'>Predicted Shelf Life</span>
                    <span class='storage-val' style='color:{risk_color};'>{s["shelf_life_days"]:.0f} days</span>
                </div>
                <div class='storage-row'>
                    <span class='storage-key'>Risk Category</span>
                    <span class='storage-val' style='color:{risk_color};'>{s["risk_level"]}</span>
                </div>
                <div class='storage-row' style='border-bottom:none;'>
                    <span class='storage-key'>Grade Classification</span>
                    <span class='storage-val'>{grade_text}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Grade Summary + Recommendation
        sum_c1, sum_c2 = st.columns(2)

        with sum_c1:
            st.markdown("<p class='section-header'>📋 Grade Summary</p>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='padding:16px; background:#F8FAFC; border-radius:10px; border:1px solid #E2E8F0;
                        font-size:13px; color:#334155; line-height:1.6;'>
                {g["summary"]}
            </div>""", unsafe_allow_html=True)

        with sum_c2:
            st.markdown("<p class='section-header'>💡 Storage Recommendation</p>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='rec-box'>
                {s["recommendation"]}
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Section 4: PDF Report ─────────────────────────────────────────────
        st.markdown("<p class='section-header'>📄 Quality Assessment Report</p>", unsafe_allow_html=True)
        pdf_c1, pdf_c2 = st.columns([2, 1])

        with pdf_c1:
            st.markdown("""
            <div style='font-size:13px; color:#64748B; line-height:1.6; padding:12px 0;'>
                Download a formatted PDF report containing the segmentation overlay, kernel metrics,
                storage projection, and grade classification for this batch.
            </div>""", unsafe_allow_html=True)

        with pdf_c2:
            report_name = f"{img_path.stem}_quality_report.pdf"
            pdf_path = global_config.reports_dir / report_name

            if st.button("🛠️ Compile PDF Report", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    try:
                        PDFReportGenerator.generate(
                            image_path=res["image_path"],
                            overlay_path=res["overlay_path"],
                            features=f,
                            grading=g,
                            storage=s,
                            output_pdf_path=pdf_path
                        )
                        st.success(f"✅ Report ready: {report_name}")
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
                        logger.exception(e)

            if pdf_path.exists():
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_file,
                        file_name=report_name,
                        mime="application/pdf",
                        use_container_width=True
                    )

        st.divider()

        # ── Section 5: Session History ─────────────────────────────────────────
        st.markdown("<p class='section-header'>🕒 Assessment History (This Session)</p>", unsafe_allow_html=True)
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No assessments run yet.")

else:
    # ── Empty State ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding:60px 40px; background:#F8FAFC;
                border-radius:16px; border:2px dashed #E2E8F0; margin-top:24px;'>
        <div style='font-size:52px; margin-bottom:12px;'>🌽</div>
        <h3 style='color:#1E293B; font-weight:600; margin:0 0 8px 0;'>Upload a Corn Image to Get Started</h3>
        <p style='color:#64748B; font-size:14px; max-width:480px; margin:0 auto;'>
            Upload a close-up .jpg photograph of a corn cob above. The system will automatically
            detect the ear, run semantic segmentation, grade the quality, and estimate shelf life.
        </p>
    </div>
    """, unsafe_allow_html=True)
