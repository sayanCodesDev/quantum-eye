"""
QUANTUM EYE — BEYOND RANDOMNESS
Machine Learning for Intelligent Complex Signal Analysis
Main Application Entrypoint
"""
import os
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="QUANTUM EYE — Beyond Randomness",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS Stylesheet
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize Session State
from modules.state_manager import init_session_state
init_session_state()

# Import Page Modules
from pages import (
    p01_overview,
    p02_upload_dataset,
    p_generator,
    p03_dataset_explorer,
    p04_signal_analysis,
    p05_statistical_analysis,
    p06_frequency_analysis,
    p07_nonlinear_dynamics,
    p08_machine_learning,
    p09_anomaly_detection,
    p10_quantum_lab,
    p11_model_comparison,
    p12_reports
)

# Sidebar Navigation & Branding
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem 0;">
        <div style="font-size: 1.5rem; font-weight: 800; color: #38bdf8; letter-spacing: -0.02em;">
            ⚛️ QUANTUM EYE
        </div>
        <div style="font-size: 0.78rem; font-weight: 700; color: #0284c7; letter-spacing: 0.12em; text-transform: uppercase;">
            BEYOND RANDOMNESS
        </div>
        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.35rem; line-height: 1.35;">
            Machine Learning for Intelligent Complex Signal Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Active Workspace Status Pill
    active_ds = st.session_state.get("dataset_name", "None")
    active_sig = st.session_state.get("signal_col", "None")
    fs_val = st.session_state.get("fs", None)
    
    st.markdown(f"""
    <div style="background-color: #1e293b; border-radius: 8px; padding: 0.75rem; border: 1px solid #334155; margin-bottom: 1.25rem;">
        <div style="font-size: 0.7rem; font-weight: 700; color: #38bdf8; text-transform: uppercase;">ACTIVE SIGNAL</div>
        <div style="font-size: 0.9rem; font-weight: 600; color: #f8fafc; word-break: break-all;">{active_sig}</div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.2rem;">
            {active_ds[:22]}...<br>
            {f'Sampling: {fs_val} Hz' if fs_val else 'Unit Index'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation Section Options
    nav_sections = [
        "1. Overview",
        "2. Upload Dataset",
        "3. Signal Generator Lab",
        "4. Dataset Explorer",
        "5. Signal Analysis",
        "6. Statistical Analysis",
        "7. Frequency Analysis",
        "8. Nonlinear Dynamics",
        "9. Machine Learning",
        "10. Anomaly Detection",
        "11. Quantum Lab",
        "12. Model Comparison",
        "13. Reports"
    ]

    selected_nav = st.radio(
        "Navigation Sections",
        nav_sections,
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("Quantum Eye v2.0 • Scientific Prototype\nLocal Workstation Mode • CPU Only")

# Page Routing
if selected_nav == "1. Overview":
    p01_overview.render()
elif selected_nav == "2. Upload Dataset":
    p02_upload_dataset.render()
elif selected_nav == "3. Signal Generator Lab":
    p_generator.render()
elif selected_nav == "4. Dataset Explorer":
    p03_dataset_explorer.render()
elif selected_nav == "5. Signal Analysis":
    p04_signal_analysis.render()
elif selected_nav == "6. Statistical Analysis":
    p05_statistical_analysis.render()
elif selected_nav == "7. Frequency Analysis":
    p06_frequency_analysis.render()
elif selected_nav == "8. Nonlinear Dynamics":
    p07_nonlinear_dynamics.render()
elif selected_nav == "9. Machine Learning":
    p08_machine_learning.render()
elif selected_nav == "10. Anomaly Detection":
    p09_anomaly_detection.render()
elif selected_nav == "11. Quantum Lab":
    p10_quantum_lab.render()
elif selected_nav == "12. Model Comparison":
    p11_model_comparison.render()
elif selected_nav == "13. Reports":
    p12_reports.render()

