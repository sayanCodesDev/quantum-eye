"""
QUANTUM EYE — Page 9: Anomaly Detection Engine
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from modules.state_manager import get_current_signal
from modules.anomaly_detection import (
    AnomalyDetector, plot_anomalies_waveform, plot_anomaly_scores
)

def render():
    st.title("🚨 Complex Signal Anomaly Detection")
    st.markdown("Isolate atypical mathematical transients, sudden distributional shifts, and rare signal shocks across deterministic, stochastic, and operational time series.")

    # Scientific Caution Alert
    st.markdown("""
    <div class="qe-callout qe-callout-warning">
        <b>Operational Precaution:</b> In scientific signal processing, an anomaly is an unusual statistical observation
        that deviates from the nominal distribution. It does <i>not</i> automatically confirm an equipment fault, physical failure,
        or security compromise without secondary physical sensors or operational confirmation.
    </div>
    """, unsafe_allow_html=True)

    time_axis, signal_values, sig_name, fs = get_current_signal()
    if signal_values is None or len(signal_values) == 0:
        st.warning("No signal loaded. Please select or upload a dataset.")
        return

    # Method Selector & Controls
    c_m, c_sens = st.columns([1.5, 1.5])
    with c_m:
        detector_name = st.selectbox(
            "Anomaly Detection Algorithm",
            [
                "Isolation Forest (Ensemble Tree Partitioning)",
                "Z-Score Thresholding (Global Normal Model)",
                "Rolling Deviation (Adaptive Local Envelope)",
                "Median Absolute Deviation (MAD - Robust)",
                "One-Class SVM (Nonlinear Support Vector Boundary)"
            ]
        )

    with c_sens:
        if "Isolation Forest" in detector_name:
            contamination = st.slider("Expected Contamination Rate", 0.01, 0.20, 0.05, 0.01)
        elif "Z-Score" in detector_name:
            z_thresh = st.slider("Z-Score Sensitivity Threshold", 1.5, 5.0, 3.0, 0.25)
        elif "Rolling" in detector_name:
            c_rw, c_rk = st.columns(2)
            with c_rw:
                roll_w = st.slider("Window Size", 10, min(500, len(signal_values)//2), 50, 10)
            with c_rk:
                roll_k = st.slider("k-Factor", 1.5, 5.0, 3.0, 0.25)
        elif "MAD" in detector_name:
            mad_thresh = st.slider("MAD Multiplier Threshold", 2.0, 6.0, 3.5, 0.25)
        else: # One-Class SVM
            oc_nu = st.slider("One-Class Nu (Upper Outlier Bound)", 0.01, 0.20, 0.05, 0.01)

    # Compute Anomaly Masks
    with st.spinner("Executing anomaly detector..."):
        if "Isolation Forest" in detector_name:
            is_anom, scores = AnomalyDetector.detect_isolation_forest(signal_values, contamination=contamination)
            method_label = f"Isolation Forest (contam={contamination})"
        elif "Z-Score" in detector_name:
            is_anom, scores = AnomalyDetector.detect_zscore(signal_values, threshold=z_thresh)
            method_label = f"Z-Score (|z| > {z_thresh})"
        elif "Rolling" in detector_name:
            is_anom, scores = AnomalyDetector.detect_rolling_deviation(signal_values, window=roll_w, k=roll_k)
            method_label = f"Rolling Deviation (w={roll_w}, k={roll_k})"
        elif "MAD" in detector_name:
            is_anom, scores = AnomalyDetector.detect_mad(signal_values, threshold=mad_thresh)
            method_label = f"Robust MAD (thresh={mad_thresh})"
        else:
            is_anom, scores = AnomalyDetector.detect_one_class_svm(signal_values, nu=oc_nu)
            method_label = f"One-Class SVM (nu={oc_nu})"

    n_anom = int(np.sum(is_anom))
    anom_pct = (n_anom / len(signal_values)) * 100

    # Save to session state for PDF report
    st.session_state.setdefault("cached_results", {})["anomaly_results"] = {
        "method": method_label,
        "anomaly_count": n_anom,
        "percentage": anom_pct
    }

    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Detected Anomalies", f"{n_anom:,}")
    col2.metric("Anomaly Rate", f"{anom_pct:.2f}%")
    col3.metric("Peak Anomaly Score", f"{np.max(scores):.3f}")
    col4.metric("Nominal Series Length", f"{len(signal_values) - n_anom:,}")

    st.markdown("---")

    # Waveform with highlighted anomalies
    t_label = f"Time ({'s' if fs else 'Sample Index'})"
    fig_anom_wave = plot_anomalies_waveform(
        time_axis=time_axis,
        signal_values=signal_values,
        is_anomaly=is_anom,
        x_label=t_label,
        method_name=method_label
    )
    st.plotly_chart(fig_anom_wave, use_container_width=True)

    # Anomaly Scores & Distribution
    col_sc1, col_sc2 = st.columns([1.5, 1.0])
    with col_sc1:
        st.subheader("Anomaly Score Timeline")
        fig_score = plot_anomaly_scores(time_axis, scores, is_anom, x_label=t_label)
        st.plotly_chart(fig_score, use_container_width=True)

    with col_sc2:
        st.subheader("Score Probability Density")
        fig_hist = go.Figure(go.Histogram(
            x=scores,
            nbinsx=35,
            marker=dict(color='#f59e0b', line=dict(color='#ffffff', width=1)),
            opacity=0.8
        ))
        fig_hist.update_layout(
            title="Anomaly Score Histogram",
            xaxis=dict(title="Score", gridcolor="#f1f5f9"),
            yaxis=dict(title="Count", gridcolor="#f1f5f9"),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            margin=dict(l=40, r=20, t=50, b=40),
            height=340
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Tabular list of anomaly events
    st.subheader(f"📋 Detected Anomalous Event Register ({n_anom} occurrences)")
    if n_anom > 0:
        anom_idx = np.where(is_anom)[0]
        anom_df = pd.DataFrame({
            "Sample Index": anom_idx,
            "Timestamp / Time": time_axis[anom_idx],
            "Signal Amplitude": np.round(signal_values[anom_idx], 4),
            "Anomaly Score": np.round(scores[anom_idx], 4)
        })
        st.dataframe(anom_df.head(100), use_container_width=True, height=250)
        st.caption(f"Showing first {min(100, n_anom)} anomalies.")
    else:
        st.success("No anomalous points detected under current sensitivity threshold.")

