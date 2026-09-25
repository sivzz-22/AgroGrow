"""
AgroGrow Interactive Streamlit Dashboard — Premium Edition.
Clean, animated UI with health status banner, no-corn detection,
and a floating Gemini-powered chatbot.
"""

import tempfile
import sys
from pathlib import Path
import importlib.util

# ── Dynamic module resolution: Ensure 'AgroGrow' is always resolvable on Streamlit Cloud & local ──
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR.parent))

if "AgroGrow" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "AgroGrow",
        str(ROOT_DIR / "__init__.py"),
        submodule_search_locations=[str(ROOT_DIR)]
    )
    if spec and spec.loader:
        agro_mod = importlib.util.module_from_spec(spec)
        sys.modules["AgroGrow"] = agro_mod
        spec.loader.exec_module(agro_mod)

import streamlit as st
import numpy as np
import cv2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

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

# ── Session State Initialisation ─────────────────────────────────────────────
if "selected_theme" not in st.session_state:
    st.session_state["selected_theme"] = "☀️ Clean Light Mode"

if "selected_engine" not in st.session_state:
    st.session_state["selected_engine"] = "🎯 Pure Trained CornNet (Recommended)"

if "selected_workflow" not in st.session_state:
    st.session_state["selected_workflow"] = "paper"
# Migrate old emoji-string values from previous sessions
elif st.session_state["selected_workflow"] not in ("paper", "multi"):
    st.session_state["selected_workflow"] = "paper"

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

# ── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    is_dark = (st.session_state.get("selected_theme") == "🌙 Dark Modern Mode")
    is_paper_mode = st.session_state.get("selected_workflow", "paper") == "paper"

    # ── Minimal header: Title + single theme icon button ──────────────────────
    _hc1, _hc2 = st.columns([6, 1])
    with _hc1:
        st.markdown("<span class='ag-sidebar-title'>🌽 AgroGrow</span>", unsafe_allow_html=True)
    with _hc2:
        _theme_icon = "🌙" if is_dark else "☀️"
        _theme_tip  = "Switch to Light Mode" if is_dark else "Switch to Dark Mode"
        if st.button(_theme_icon, key="btn_theme_toggle", help=_theme_tip):
            st.session_state["selected_theme"] = (
                "☀️ Clean Light Mode" if is_dark else "🌙 Dark Modern Mode"
            )
            st.rerun()

    st.caption("Post-Harvest Corn Quality & Storage AI")
    st.markdown("---")

    # ── Storage Parameters ────────────────────────────────────────────────────
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

# ── Resolve mode variables (needed before theme CSS block) ────────────────────
is_paper_mode = st.session_state.get("selected_workflow", "paper") == "paper"
is_pure_model = True
corn_variety_input = "🌽 Dent Corn (Yellow/White) — Commercial"

# ── Dynamic Theme Injection (Guarantees Perfect Contrast for Chosen Mode) ────
if is_dark:
    theme_css = """
    :root {
        --ag-bg-app: #0e1117;
        --ag-bg-card: rgba(30, 41, 59, 0.90);
        --ag-bg-surface: #1e293b;
        --ag-bg-subtle: #0f172a;
        --ag-border: rgba(255, 255, 255, 0.12);
        --ag-border-subtle: rgba(255, 255, 255, 0.08);
        --ag-text-primary: #f8fafc;
        --ag-text-secondary: #e2e8f0;
        --ag-text-muted: #94a3b8;
        --ag-text-faint: #64748b;
        --ag-popover-bg: #0f172a;
        --ag-popover-border: #334155;
        --ag-pbar-track: #334155;
        
        --ag-variety-bg: rgba(22, 101, 52, 0.30);
        --ag-variety-border: #22c55e;
        --ag-variety-text: #86efac;
        
        --ag-rec-bg: rgba(180, 83, 9, 0.22);
        --ag-rec-border: rgba(245, 158, 11, 0.40);
        --ag-rec-text: #fde68a;
        
        --ag-empty-bg: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.85) 100%);
        --ag-empty-border: #334155;

        --ag-hb-healthy-bg: linear-gradient(135deg, rgba(22, 101, 52, 0.35), rgba(20, 83, 45, 0.50));
        --ag-hb-healthy-border: #22c55e;
        --ag-hb-healthy-title: #86efac;
        --ag-hb-healthy-desc: #bbf7d0;
        --ag-hb-healthy-shadow: rgba(34, 197, 94, 0.20);

        --ag-hb-warning-bg: linear-gradient(135deg, rgba(161, 98, 7, 0.35), rgba(133, 77, 14, 0.50));
        --ag-hb-warning-border: #f59e0b;
        --ag-hb-warning-title: #fde68a;
        --ag-hb-warning-desc: #fef08a;
        --ag-hb-warning-shadow: rgba(245, 158, 11, 0.20);

        --ag-hb-danger-bg: linear-gradient(135deg, rgba(185, 28, 28, 0.35), rgba(153, 27, 27, 0.50));
        --ag-hb-danger-border: #ef4444;
        --ag-hb-danger-title: #fca5a5;
        --ag-hb-danger-desc: #fecaca;
        --ag-hb-danger-shadow: rgba(239, 68, 68, 0.20);

        --ag-hb-nocorn-bg: linear-gradient(135deg, rgba(51, 65, 85, 0.40), rgba(30, 41, 59, 0.60));
        --ag-hb-nocorn-border: #94a3b8;
        --ag-hb-nocorn-title: #f1f5f9;
        --ag-hb-nocorn-desc: #cbd5e1;
        --ag-hb-nocorn-shadow: rgba(148, 163, 184, 0.15);
    }
    .stApp {
        background-color: #0e1117 !important;
        color: #f8fafc !important;
    }
    .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp div, .stApp li, .stApp td, .stApp th {
        color: #f8fafc !important;
    }
    /* Fix Streamlit internal text elements in dark mode */
    .stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown td,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li {
        color: #f0f4f8 !important;
    }
    /* Fix widget labels */
    .stSelectbox label, .stSlider label, .stRadio label,
    .stTextInput label, .stFileUploader label, .stExpander label,
    [data-testid="stWidgetLabel"] { color: #e2e8f0 !important; }
    /* Fix selectbox/dropdown text */
    [data-testid="stSelectbox"] div, [data-testid="stSelectbox"] span { color: #f8fafc !important; }
    /* Fix expander */
    [data-testid="stExpander"] summary, [data-testid="stExpander"] p { color: #f0f4f8 !important; }
    /* Fix metric values */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #f8fafc !important; }
    /* Fix slider values */
    [data-testid="stSlider"] p, [data-testid="stSlider"] span { color: #e2e8f0 !important; }
    /* Fix caption / small text */
    .stCaption, small { color: #94a3b8 !important; }
    /* Fix st.success / st.info / st.warning banners */
    [data-testid="stAlert"] p, [data-testid="stAlert"] span { color: inherit !important; }
    /* Fix table cells */
    [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th { color: #f0f4f8 !important; }
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.10) !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] p[data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] .stCaption {
        color: #94a3b8 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(30, 41, 59, 0.6) !important;
        border: 2px dashed rgba(255, 255, 255, 0.15) !important;
        border-radius: 14px !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #cbd5e1 !important;
    }
    [data-testid="stChatMessage"] {
        background-color: #1e293b !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
    }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] span {
        color: #f0f4f8 !important;
    }
    /* Fix upload button text */
    [data-testid="stFileUploaderDropzoneInput"] + div span { color: #cbd5e1 !important; }
    /* Fix segmented control text */
    [data-testid="stSegmentedControl"] span { color: #e2e8f0 !important; }
    /* Section headers and framing headers in dark */
    .section-hdr, .framing-hdr { color: #f0f4f8 !important; }
    /* Grade badges fix for dark */
    .grade-A { background: rgba(22,101,52,0.4) !important; color: #86efac !important; }
    .grade-B { background: rgba(30,64,175,0.4) !important; color: #93c5fd !important; }
    .grade-C { background: rgba(133,77,14,0.4) !important; color: #fde68a !important; }
    .grade-D { background: rgba(153,27,27,0.4) !important; color: #fca5a5 !important; }

    /* ── Dark mode: ALL secondary (inactive) buttons — auto dark background ── */
    [data-testid="stBaseButton-secondary"] {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border-color: rgba(255,255,255,0.15) !important;
    }
    [data-testid="stBaseButton-secondary"]:hover {
        background-color: #273548 !important;
        color: #f8fafc !important;
        border-color: rgba(255,255,255,0.25) !important;
    }
    /* ── Dark mode: primary (active) buttons keep green ── */
    [data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #15803d 0%, #16a34a 50%, #059669 100%) !important;
        color: #ffffff !important;
        border-color: #16a34a !important;
    }
    [data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(135deg, #166534 0%, #15803d 50%, #047857 100%) !important;
    }
    /* ── Dark mode: sidebar theme-toggle icon button exception ── */
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
        background: transparent !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover,
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
        background: rgba(255,255,255,0.08) !important;
        border-color: #16a34a !important;
    }
    /* ── Dark mode: segmented control buttons ── */
    [data-testid="stSegmentedControl"] button {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stSegmentedControl"] button[aria-selected="true"] {
        background-color: #334155 !important;
        color: #f8fafc !important;
    }
    /* ── Dark mode: expander ── */
    [data-testid="stExpander"] {
        background-color: rgba(30,41,59,0.6) !important;
        border-color: rgba(255,255,255,0.12) !important;
    }
    /* ── Dark mode: selectbox dropdown ── */
    [data-testid="stSelectbox"] > div > div {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border-color: rgba(255,255,255,0.15) !important;
    }
    """
else:
    theme_css = """
    :root {
        --ag-bg-app: #f8fafc;
        --ag-bg-card: #ffffff;
        --ag-bg-surface: #ffffff;
        --ag-bg-subtle: #f1f5f9;
        --ag-border: #e2e8f0;
        --ag-border-subtle: #f1f5f9;
        --ag-text-primary: #0f172a;
        --ag-text-secondary: #334155;
        --ag-text-muted: #64748b;
        --ag-text-faint: #94a3b8;
        --ag-popover-bg: #ffffff;
        --ag-popover-border: #cbd5e1;
        --ag-pbar-track: #e2e8f0;
        
        --ag-variety-bg: #f0fdf4;
        --ag-variety-border: #16a34a;
        --ag-variety-text: #166534;
        
        --ag-rec-bg: #fffbeb;
        --ag-rec-border: #fde68a;
        --ag-rec-text: #92400e;
        
        --ag-empty-bg: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        --ag-empty-border: #cbd5e1;

        --ag-hb-healthy-bg: linear-gradient(135deg, #f0fdf4, #dcfce7);
        --ag-hb-healthy-border: #16a34a;
        --ag-hb-healthy-title: #166534;
        --ag-hb-healthy-desc: #15803d;
        --ag-hb-healthy-shadow: rgba(22, 163, 74, 0.12);

        --ag-hb-warning-bg: linear-gradient(135deg, #fffbeb, #fef9c3);
        --ag-hb-warning-border: #d97706;
        --ag-hb-warning-title: #92400e;
        --ag-hb-warning-desc: #b45309;
        --ag-hb-warning-shadow: rgba(217, 119, 6, 0.12);

        --ag-hb-danger-bg: linear-gradient(135deg, #fff1f2, #fde8e8);
        --ag-hb-danger-border: #dc2626;
        --ag-hb-danger-title: #991b1b;
        --ag-hb-danger-desc: #b91c1c;
        --ag-hb-danger-shadow: rgba(220, 38, 38, 0.12);

        --ag-hb-nocorn-bg: linear-gradient(135deg, #f8fafc, #f1f5f9);
        --ag-hb-nocorn-border: #64748b;
        --ag-hb-nocorn-title: #334155;
        --ag-hb-nocorn-desc: #475569;
        --ag-hb-nocorn-shadow: rgba(100, 116, 139, 0.10);
    }
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #0f172a;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] p[data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] .stCaption {
        color: #64748b !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #f1f5f9 !important;
        border: 2px dashed #cbd5e1 !important;
        border-radius: 14px !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #334155 !important;
    }
    [data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
    }
    """

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
}}

{theme_css}

/* ── Sidebar compact title ── */
.ag-sidebar-title {{
    font-size: 17px;
    font-weight: 800;
    color: var(--ag-text-primary);
    white-space: nowrap;
    display: inline-block;
    line-height: 1.6;
}}

/* ── Theme toggle icon button in sidebar — tiny, no border ── */
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {{
    padding: 2px 4px !important;
    min-height: unset !important;
    height: 28px !important;
    width: 28px !important;
    font-size: 16px !important;
    line-height: 1 !important;
    border-radius: 6px !important;
    background: transparent !important;
    border: 1px solid var(--ag-border) !important;
    box-shadow: none !important;
    color: var(--ag-text-primary) !important;
}}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover,
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {{
    background: var(--ag-bg-subtle) !important;
    border-color: #16a34a !important;
}}

/* ── Mode selector card-buttons ── */
button.ag-mode-card-btn {{
    min-height: 100px !important;
    white-space: pre-line !important;
    border-radius: 14px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 18px 16px !important;
    line-height: 1.7 !important;
    letter-spacing: 0 !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0 !important;
    cursor: pointer !important;
}}
button.ag-mode-card-btn:hover {{
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 28px rgba(0,0,0,0.12) !important;
}}
/* Inactive card */
button.ag-mode-card-btn[data-testid="stBaseButton-secondary"] {{
    background: var(--ag-bg-surface) !important;
    border: 1.5px solid var(--ag-border) !important;
    color: var(--ag-text-secondary) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}}
/* Active card (green glow) */
button.ag-mode-card-btn[data-testid="stBaseButton-primary"] {{
    background: linear-gradient(135deg, rgba(22,163,74,0.12) 0%, rgba(5,150,105,0.08) 100%) !important;
    border: 2px solid #16a34a !important;
    color: var(--ag-text-primary) !important;
    box-shadow: 0 4px 20px rgba(22,163,74,0.18) !important;
}}

/* ── Animated gradient header ── */
.ag-header {{
    background: linear-gradient(135deg, #0f4c2a 0%, #1a6b3a 40%, #0d5c3e 70%, #1a4a2a 100%);
    background-size: 300% 300%;
    animation: gradientShift 8s ease infinite;
    border-radius: 20px;
    padding: 36px 32px 32px 32px;
    margin-bottom: 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
}}
.ag-header::before {{
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.04) 0%, transparent 60%);
    animation: rotate 15s linear infinite;
}}
@keyframes gradientShift {{
    0%   {{ background-position: 0% 50%; }}
    50%  {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}
@keyframes rotate {{
    from {{ transform: rotate(0deg); }}
    to   {{ transform: rotate(360deg); }}
}}
.ag-title {{
    font-size: clamp(24px, 5vw, 40px);
    font-weight: 800;
    color: #ffffff !important;
    margin: 0;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 12px rgba(0,0,0,0.3);
}}
.ag-subtitle {{
    color: rgba(255,255,255,0.85) !important;
    font-size: clamp(12px, 2vw, 15px);
    margin: 8px 0 0 0;
    font-weight: 400;
}}
.ag-pill {{
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 999px;
    padding: 4px 16px;
    font-size: clamp(10px, 1.5vw, 12px);
    color: rgba(255,255,255,0.95) !important;
    font-weight: 600;
    margin-top: 14px;
    letter-spacing: 1px;
    text-transform: uppercase;
    backdrop-filter: blur(8px);
}}

/* ── Health status banner ── */
.health-banner {{
    border-radius: 14px;
    padding: 18px 24px;
    margin: 16px 0;
    display: flex;
    align-items: center;
    gap: 16px;
    animation: slideDown 0.4s ease;
    flex-wrap: wrap;
}}
@keyframes slideDown {{
    from {{ opacity: 0; transform: translateY(-12px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.health-banner-healthy {{
    background: var(--ag-hb-healthy-bg);
    border: 1.5px solid var(--ag-hb-healthy-border);
    box-shadow: 0 4px 20px var(--ag-hb-healthy-shadow);
}}
.health-banner-healthy .health-title {{ color: var(--ag-hb-healthy-title) !important; }}
.health-banner-healthy .health-desc {{ color: var(--ag-hb-healthy-desc) !important; }}

.health-banner-warning {{
    background: var(--ag-hb-warning-bg);
    border: 1.5px solid var(--ag-hb-warning-border);
    box-shadow: 0 4px 20px var(--ag-hb-warning-shadow);
}}
.health-banner-warning .health-title {{ color: var(--ag-hb-warning-title) !important; }}
.health-banner-warning .health-desc {{ color: var(--ag-hb-warning-desc) !important; }}

.health-banner-danger {{
    background: var(--ag-hb-danger-bg);
    border: 1.5px solid var(--ag-hb-danger-border);
    box-shadow: 0 4px 20px var(--ag-hb-danger-shadow);
}}
.health-banner-danger .health-title {{ color: var(--ag-hb-danger-title) !important; }}
.health-banner-danger .health-desc {{ color: var(--ag-hb-danger-desc) !important; }}

.health-banner-nocorn {{
    background: var(--ag-hb-nocorn-bg);
    border: 1.5px solid var(--ag-hb-nocorn-border);
    box-shadow: 0 4px 20px var(--ag-hb-nocorn-shadow);
}}
.health-banner-nocorn .health-title {{ color: var(--ag-hb-nocorn-title) !important; }}
.health-banner-nocorn .health-desc {{ color: var(--ag-hb-nocorn-desc) !important; }}

.health-icon {{ font-size: clamp(28px, 5vw, 40px); flex-shrink: 0; }}
.health-title {{ font-size: clamp(14px, 2.5vw, 18px); font-weight: 700; margin: 0; }}
.health-desc  {{ font-size: clamp(11px, 1.8vw, 13px); margin: 3px 0 0 0; opacity: 0.9; }}

/* ── Glass KPI cards ── */
.kpi-card {{
    background: var(--ag-bg-card);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: clamp(14px, 2vw, 22px) clamp(10px, 1.5vw, 18px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.06), 0 8px 32px rgba(0,0,0,0.04);
    border: 1px solid var(--ag-border);
    border-top: 4px solid #2B6CB0;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    animation: fadeUp 0.5s ease both;
    height: 100%;
}}
.kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.12), 0 16px 48px rgba(0,0,0,0.08);
}}
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(16px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.kpi-label {{
    font-size: clamp(9px, 1.2vw, 10.5px);
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--ag-text-faint);
    margin-bottom: 8px;
}}
.kpi-value {{
    font-size: clamp(22px, 4vw, 32px);
    font-weight: 800;
    color: var(--ag-text-primary);
    line-height: 1.1;
    margin-bottom: 6px;
}}
.kpi-sub {{ font-size: 12px; color: var(--ag-text-muted); }}

/* ── Grade badge ── */
.grade-badge {{
    display: inline-block;
    padding: 3px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    margin-top: 6px;
}}
.grade-A {{ background:#dcfce7; color:#166534; }}
.grade-B {{ background:#dbeafe; color:#1e40af; }}
.grade-C {{ background:#fef9c3; color:#854d0e; }}
.grade-D {{ background:#fee2e2; color:#991b1b; }}

/* ── Section header ── */
.section-hdr {{
    font-size: clamp(13px, 2vw, 15px);
    font-weight: 700;
    color: var(--ag-text-primary);
    margin: 0 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--ag-border-subtle);
    display: flex;
    align-items: center;
    gap: 6px;
}}

/* ── Framing header ── */
.framing-hdr {{
    font-size: 14px;
    font-weight: 700;
    color: var(--ag-text-primary);
    margin: 14px 0 6px 0;
}}

/* ── Variety chip ── */
.variety-chip {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--ag-variety-bg);
    border: 1.5px solid var(--ag-variety-border);
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 13px;
    color: var(--ag-variety-text);
    font-weight: 500;
    margin-bottom: 16px;
    animation: fadeUp 0.4s ease;
    flex-wrap: wrap;
}}

/* ── Progress bars ── */
.pbar-wrap {{ margin-bottom: 14px; }}
.pbar-meta {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
    font-size: 13px;
    font-weight: 500;
    color: var(--ag-text-secondary);
    flex-wrap: wrap;
    gap: 4px;
}}
.pbar-track {{
    background: var(--ag-pbar-track);
    border-radius: 999px;
    height: 9px;
    overflow: hidden;
}}
.pbar-fill {{
    height: 9px;
    border-radius: 999px;
    transition: width 0.6s ease;
}}

/* ── Storage card ── */
.st-card {{
    background: var(--ag-bg-surface);
    border: 1px solid var(--ag-border);
    border-radius: 14px;
    padding: 20px;
}}
.st-row {{
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid var(--ag-border-subtle);
    font-size: 13px;
    flex-wrap: wrap;
    gap: 4px;
}}
.st-key {{ color: var(--ag-text-muted); font-weight: 500; }}
.st-val {{ color: var(--ag-text-primary); font-weight: 600; }}

/* ── Summary & Legend Boxes ── */
.summary-box {{
    padding: 16px;
    background: var(--ag-bg-surface);
    border-radius: 12px;
    border: 1px solid var(--ag-border);
    font-size: 13px;
    color: var(--ag-text-secondary);
    line-height: 1.65;
}}

.inline-legend {{
    display: flex;
    gap: 10px;
    justify-content: center;
    align-items: center;
    padding: 9px 14px;
    background: var(--ag-bg-surface);
    border-radius: 10px;
    border: 1px solid var(--ag-border);
    margin-top: 8px;
    font-size: 12px;
    font-weight: 600;
    color: var(--ag-text-secondary);
    flex-wrap: wrap;
}}

/* ── Recommendation box ── */
.rec-box {{
    background: var(--ag-rec-bg);
    border: 1px solid var(--ag-rec-border);
    border-radius: 12px;
    padding: 16px;
    font-size: 13px;
    color: var(--ag-rec-text);
    line-height: 1.65;
}}

/* ── Empty state ── */
.empty-state {{
    text-align: center;
    padding: clamp(40px, 8vw, 72px) clamp(20px, 5vw, 40px);
    background: var(--ag-empty-bg);
    border-radius: 20px;
    border: 2px dashed var(--ag-empty-border);
    margin-top: 20px;
    animation: fadeUp 0.6s ease;
}}
.empty-state h3 {{
    color: var(--ag-text-primary) !important;
}}
.empty-state p {{
    color: var(--ag-text-muted) !important;
}}

/* ── Primary Action Button (Run Quality Assessment) ── */
button[kind="primary"] {{
    background: linear-gradient(135deg, #15803d 0%, #16a34a 50%, #059669 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-size: clamp(13px, 2vw, 16px) !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px !important;
    padding: 12px 24px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 14px rgba(22, 163, 74, 0.35) !important;
    transition: all 0.2s ease !important;
    margin-top: 10px !important;
    margin-bottom: 15px !important;
}}
button[kind="primary"]:hover {{
    background: linear-gradient(135deg, #166534 0%, #15803d 50%, #047857 100%) !important;
    box-shadow: 0 6px 20px rgba(22, 163, 74, 0.5) !important;
    transform: translateY(-1px) !important;
}}

/* ── Floating AgroGrow Chatbot Button (Fixed at Bottom-Right Corner) ── */
div[data-testid="stPopover"] {{
    position: fixed !important;
    bottom: 28px !important;
    right: 28px !important;
    left: auto !important;
    width: auto !important;
    max-width: fit-content !important;
    z-index: 999999 !important;
}}

div[data-testid="stPopover"] > button {{
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
}}

div[data-testid="stPopover"] > button:hover {{
    transform: translateY(-3px) scale(1.04) !important;
    box-shadow: 0 10px 32px rgba(22, 163, 74, 0.65) !important;
    background: linear-gradient(135deg, #15803d 0%, #047857 100%) !important;
}}

/* Floating popover dialog window */
div[data-testid="stPopoverBody"] {{
    max-width: 480px !important;
    min-width: min(360px, 90vw) !important;
    max-height: 80vh !important;
    border-radius: 18px !important;
    box-shadow: 0 16px 48px rgba(0,0,0,0.22) !important;
    border: 1.5px solid var(--ag-popover-border) !important;
    padding: 16px !important;
    background: var(--ag-popover-bg) !important;
    color: var(--ag-text-primary) !important;
    overflow-y: auto !important;
}}

/* ── Responsive: Tablet (≤900px) ── */
@media (max-width: 900px) {{
    .ag-header {{
        padding: 24px 18px 20px 18px;
        border-radius: 14px;
    }}
    .kpi-card {{ padding: 14px 12px; }}
    .health-banner {{ padding: 14px 16px; gap: 10px; }}
    .empty-state {{ padding: 44px 24px; }}
}}

/* ── Responsive: Mobile (≤600px) ── */
@media (max-width: 600px) {{
    .ag-header {{
        padding: 18px 12px 16px 12px;
        border-radius: 10px;
        margin-bottom: 16px;
    }}
    .kpi-card {{ padding: 12px 10px; border-radius: 12px; }}
    .kpi-value {{ font-size: 24px; }}
    .health-banner {{
        padding: 12px 14px;
        gap: 8px;
        flex-direction: column;
        align-items: flex-start;
    }}
    .health-icon {{ font-size: 28px; }}
    .health-title {{ font-size: 15px; }}
    .health-desc {{ font-size: 12px; }}
    .variety-chip {{ padding: 8px 12px; font-size: 12px; }}
    .section-hdr {{ font-size: 13px; }}
    .rec-box {{ padding: 12px; font-size: 12px; }}
    .st-card {{ padding: 14px; }}
    .st-row {{ font-size: 12px; }}
    .summary-box {{ font-size: 12px; }}
    div[data-testid="stPopover"] {{
        bottom: 16px !important;
        right: 16px !important;
    }}
    div[data-testid="stPopover"] > button {{
        padding: 10px 18px !important;
        font-size: 13px !important;
    }}
    /* Stack columns vertically on mobile */
    [data-testid="stColumns"] > div {{
        min-width: 100% !important;
    }}
    button.ag-mode-card-btn {{
        min-height: 70px !important;
        font-size: 13px !important;
        padding: 12px 10px !important;
    }}
}}

/* ── Responsive: Extra Small (≤400px) ── */
@media (max-width: 400px) {{
    .ag-title {{ font-size: 22px; }}
    .ag-subtitle {{ font-size: 11px; }}
    .kpi-value {{ font-size: 20px; }}
    .kpi-label {{ font-size: 9px; }}
}}

/* ── Main content area: fluid max-width ── */
.block-container {{
    max-width: 1200px !important;
    padding-left: clamp(8px, 2.5vw, 3rem) !important;
    padding-right: clamp(8px, 2.5vw, 3rem) !important;
    padding-top: 1rem !important;
}}

/* ── Images fully fluid ── */
[data-testid="stImage"] img {{
    max-width: 100% !important;
    height: auto !important;
    border-radius: 10px;
}}

/* ── Sidebar collapse gracefully on mobile ── */
[data-testid="stSidebar"] {{
    min-width: 200px !important;
}}
@media (max-width: 768px) {{
    [data-testid="stSidebar"] {{
        min-width: 0 !important;
    }}
    .block-container {{
        padding-left: 8px !important;
        padding-right: 8px !important;
    }}
}}
</style>
""", unsafe_allow_html=True)



# ── Animated Header ───────────────────────────────────────────────────────────
st.markdown("""
<div class='ag-header'>
  <p class='ag-title'>🌽 AgroGrow</p>
  <p class='ag-subtitle'>Post-Harvest Corn Quality Assessment &amp; Shelf-Life Forecasting</p>
  <span class='ag-pill'>Semantic Segmentation · AI Grading · Storage Prediction</span>
</div>
""", unsafe_allow_html=True)

# ── Mode Selector (in main page) ──────────────────────────────────────────────
_cur_mode = st.session_state.get("selected_workflow", "paper")
_mc1, _mc2 = st.columns(2)
with _mc1:
    _paper_active = (_cur_mode == "paper")
    if st.button(
        "🌾  Zea Mays (CornNet)",
        key="btn_mode_paper",
        use_container_width=True,
        type="primary" if _paper_active else "secondary"
    ):
        st.session_state["selected_workflow"] = "paper"
        st.session_state["current_analysis"] = None
        st.rerun()
with _mc2:
    _multi_active = (_cur_mode == "multi")
    if st.button(
        "🔬  Multi Variety (AgroGrow)",
        key="btn_mode_multi",
        use_container_width=True,
        type="primary" if _multi_active else "secondary"
    ):
        st.session_state["selected_workflow"] = "multi"
        st.session_state["current_analysis"] = None
        st.rerun()

# JS: tag mode buttons so CSS can style them as cards (runs after DOM is ready)
st.markdown("""
<script>
(function tagModeBtns() {
    var els = document.querySelectorAll(
        '[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"]'
    );
    var tagged = 0;
    els.forEach(function(el) {
        var t = el.innerText || el.textContent || "";
        if (t.includes("Zea Mays") || t.includes("Multi Variety")) {
            el.classList.add("ag-mode-card-btn");
            tagged++;
        }
    });
    if (tagged < 2) { setTimeout(tagModeBtns, 150); }
})();
</script>
""", unsafe_allow_html=True)

# ── Resolve mode-specific settings ────────────────────────────────────────────
is_paper_mode = (st.session_state.get("selected_workflow", "paper") == "paper")
if is_paper_mode:
    is_pure_model = True
    corn_variety_input = "🌽 Dent Corn (Yellow/White) — Commercial"
    _paper_weights = global_config.weights_dir / "paper_model.pth"
    if _paper_weights.exists():
        st.markdown(
            "<div style='background:rgba(34,197,94,0.1); border:1px solid rgba(34,197,94,0.3); "
            "border-radius:10px; padding:10px 16px; margin:8px 0; font-size:13px; color:inherit;'>"
            "🌾 <b>Research Paper Mode Active:</b> Dedicated single-variety model "
            "<code>weights/paper_model.pth</code> is loaded. Pure deep learning segmentation "
            "(no heuristic filters or variety overrides).</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background:rgba(234,179,8,0.1); border:1px solid rgba(234,179,8,0.3); "
            "border-radius:10px; padding:10px 16px; margin:8px 0; font-size:13px; color:inherit;'>"
            "🌾 <b>Research Paper Mode Active:</b> Pure single-variety CornNet pipeline. "
            "<i>(Dedicated weights <code>weights/paper_model.pth</code> not yet trained — "
            "currently using <code>best_model.pth</code>. Run <code>python train_paper.py</code> to train it.)</i></div>",
            unsafe_allow_html=True
        )
else:
    # Multi-Variety: show compact inline controls
    with st.expander("⚙️ Multi-Variety Settings", expanded=False):
        _engine = st.radio(
            "Inference Engine",
            ["🎯 CornNet (Recommended)", "🔬 Heuristic Filtered (Legacy)"],
            index=0,
            horizontal=True,
            key="mv_engine"
        )
        is_pure_model = (_engine == "🎯 CornNet (Recommended)")
        corn_variety_input = st.selectbox(
            "Corn Variety",
            [
                "🔍 Auto-Detect Variety",
                "🌽 Dent Corn (Yellow/White) — Commercial",
                "🎨 Indian / Flint Corn (Multicoloured)",
                "🍬 Sweet Corn (Pale Cream/Yellow)",
                "🍿 Popcorn (Small Hard Kernels)",
                "🔵 Blue / Black Corn (Hopi variety)",
            ],
            index=0,
            key="mv_variety"
        )

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

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
    st.markdown("<p class='framing-hdr'>✂️ Cob Framing Mode</p>", unsafe_allow_html=True)
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
                predictor = CornPredictor(device=use_device, paper_mode=is_paper_mode)
                mask, overlay, _, confidence = predictor.predict_single(
                    inference_img_path, auto_crop=False,
                    corn_variety=corn_variety_input,
                    pure_model=is_pure_model
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

                if is_paper_mode:
                    target_variety = "Commercial Fresh Sweet Corn (Zea mays) — Research Paper Standard"
                else:
                    target_variety = getattr(predictor, "last_detected_variety",
                                           "Commercial Dent Corn (Zea mays indentata)")

                st.session_state.current_analysis = {
                    "image_path":   img_path,
                    "overlay_path": overlay_temp_path,
                    "features":     features,
                    "grading":      grading,
                    "variety":      target_variety,
                    "is_paper_mode": is_paper_mode,
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
        is_single_var = res.get("is_paper_mode", False)

        # Build chatbot context from current results
        chat_context = {
            **f,
            **g,
            **s,
            "variety": variety_name
        }

        st.divider()

        # Variety chip
        chip_label = "Target Crop:" if is_single_var else "Detected Variety:"
        var_emoji = "🌽"
        if not is_single_var:
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
            <span><strong>{chip_label}</strong> {variety_name}</span>
        </div>""", unsafe_allow_html=True)

        # ── Health Status Banner ────────────────────────────────────────────────
        no_corn = f.get("no_corn_detected", False)
        q = f["quality_score"]
        d = f["disease_percentage"]

        if no_corn:
            st.markdown("""
            <div class='health-banner health-banner-nocorn'>
                <div class='health-icon'>🔍</div>
                <div>
                    <h3 class='health-title'>No Corn Detected in Image</h3>
                    <p class='health-desc'>The uploaded image does not appear to contain a valid corn cob or ear. Please upload a clear photo of corn.</p>
                </div>
            </div>""", unsafe_allow_html=True)
        elif q >= 75 and d < 5.0:
            st.markdown(f"""
            <div class='health-banner health-banner-healthy'>
                <div class='health-icon'>✅</div>
                <div>
                    <h3 class='health-title'>Corn Batch is Healthy</h3>
                    <p class='health-desc'>Overall quality score is <strong>{q:.1f}/100</strong> with minimal disease ({d:.1f}%). Safe for standard or extended storage.</p>
                </div>
            </div>""", unsafe_allow_html=True)
        elif q >= 45 and d < 20.0:
            st.markdown(f"""
            <div class='health-banner health-banner-warning'>
                <div class='health-icon'>⚠️</div>
                <div>
                    <h3 class='health-title'>Moderate Defects Detected</h3>
                    <p class='health-desc'>Quality score is <strong>{q:.1f}/100</strong> with {d:.1f}% diseased/damaged kernels. Immediate aeration or drying recommended.</p>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='health-banner health-banner-danger'>
                <div class='health-icon'>❌</div>
                <div>
                    <h3 class='health-title'>Severe Disease / Rotten Corn Detected</h3>
                    <p class='health-desc'>Quality score is critical at <strong>{q:.1f}/100</strong> with {d:.1f}% diseased kernels. Not recommended for long-term storage or commercial sale.</p>
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
            <div class='inline-legend'>
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
            <div class='summary-box'>
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
        <h3 style='font-weight:700;margin:0 0 10px 0;font-size:22px;'>
            Upload a Corn Image to Get Started
        </h3>
        <p style='font-size:14px;max-width:520px;margin:0 auto;line-height:1.7;'>
            Upload a photo of a corn cob in any standard image format (.jpg, .jpeg, .png, .webp, .bmp).
            The system will automatically detect the ear, run semantic segmentation,
            classify the grain variety, evaluate USDA/ISO grade, and forecast shelf life.
        </p>
        <div style='display:flex;gap:16px;justify-content:center;margin-top:24px;flex-wrap:wrap;'>
            <div style='background:var(--ag-bg-card);border:1px solid var(--ag-border);border-radius:12px;padding:14px 20px;
                        font-size:13px;color:var(--ag-text-secondary);box-shadow:0 2px 6px rgba(0,0,0,0.04);'>🔬 Semantic Segmentation</div>
            <div style='background:var(--ag-bg-card);border:1px solid var(--ag-border);border-radius:12px;padding:14px 20px;
                        font-size:13px;color:var(--ag-text-secondary);box-shadow:0 2px 6px rgba(0,0,0,0.04);'>📊 USDA / ISO Grading</div>
            <div style='background:var(--ag-bg-card);border:1px solid var(--ag-border);border-radius:12px;padding:14px 20px;
                        font-size:13px;color:var(--ag-text-secondary);box-shadow:0 2px 6px rgba(0,0,0,0.04);'>📦 Shelf-Life Forecast</div>
            <div style='background:var(--ag-bg-card);border:1px solid var(--ag-border);border-radius:12px;padding:14px 20px;
                        font-size:13px;color:var(--ag-text-secondary);box-shadow:0 2px 6px rgba(0,0,0,0.04);'>🌾 5 Variety Detection</div>
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
