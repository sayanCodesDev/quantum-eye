"""
QUANTUM EYE — Page 1: Overview Dashboard
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from modules.state_manager import get_current_signal
from modules.signal_analysis import compute_time_domain_metrics, plot_waveform
from modules.frequency_analysis import compute_welch_psd, compute_spectral_metrics, plot_psd_spectrum
from modules.nonlinear_analysis import compute_shannon_entropy, compute_permutation_entropy, compute_recurrence_matrix, plot_recurrence_matrix

def render():
    # Hero Banner
    st.markdown("""
    <div class="qe-hero">
        <div class="qe-hero-subtitle">BEYOND RANDOMNESS</div>
        <div class="qe-hero-title">QUANTUM EYE</div>
        <div class="qe-hero-desc">
            An interactive scientific laboratory for complex signal analysis, mathematical pattern discovery,
            nonlinear dynamics, machine learning intelligence, and experimental quantum feature mapping.
        </div>
        <div style="margin-top: 1rem;">
            <span class="qe-badge qe-badge-cyan">Signal Processing</span>
            <span class="qe-badge qe-badge-teal">Nonlinear Dynamics</span>
            <span class="qe-badge qe-badge-violet">Quantum ML</span>
            <span class="qe-badge qe-badge-amber">Anomaly Intelligence</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    time_axis, signal_values, sig_name, fs = get_current_signal()

    if signal_values is None or len(signal_values) == 0:
        st.warning("No active signal loaded. Please navigate to 'Upload Dataset' or choose a benchmark dataset.")
        return

    # Real-Time Operational KPI Metrics
    td = compute_time_domain_metrics(signal_values)
    freqs, psd, freq_unit = compute_welch_psd(signal_values, fs=fs)
    spec_meta = compute_spectral_metrics(freqs, psd, freq_unit)
    shannon = compute_shannon_entropy(signal_values)
    _, norm_pe = compute_permutation_entropy(signal_values, m=3, tau=1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Active Signal",
            value=f"{sig_name[:14]}...",
            delta=f"{st.session_state.get('dataset_name', '')[:20]}"
        )
    with c2:
        st.metric(
            label="Sample Volume",
            value=f"{len(signal_values):,}",
            delta=f"fs: {fs} Hz" if fs else "Unit Samples"
        )
    with c3:
        dom_f = spec_meta.get("dominant_frequency", 0.0)
        st.metric(
            label="Dominant Frequency",
            value=f"{dom_f:.2f} {freq_unit}",
            delta=f"Centroid: {spec_meta.get('spectral_centroid', 0.0):.2f}"
        )
    with c4:
        st.metric(
            label="Complexity (Permutation Entropy)",
            value=f"{norm_pe:.3f}",
            delta=f"Shannon: {shannon:.2f} bits"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Overview Visualizations: Signal Waveform & Spectral Density
    col_w, col_s = st.columns([1.2, 1.0])
    with col_w:
        st.subheader("Signal Waveform")
        fig_w = plot_waveform(
            time_axis=time_axis,
            signal_values=signal_values,
            title=f"Active Series: {sig_name}",
            signal_name=sig_name,
            x_label=f"Time ({'s' if fs else 'Sample Index'})",
            y_label="Amplitude"
        )
        st.plotly_chart(fig_w, use_container_width=True)

    with col_s:
        st.subheader("Welch Power Spectral Density")
        fig_s = plot_psd_spectrum(freqs, psd, freq_unit=freq_unit, log_scale=True)
        st.plotly_chart(fig_s, use_container_width=True)

    # Recurrence & Quick Laboratory Status
    c_rec, c_stat = st.columns([1, 1.2])
    with c_rec:
        st.subheader("Phase Recurrence Topology")
        r_mat, rqa_meta = compute_recurrence_matrix(signal_values[:min(len(signal_values), 1200)], m=3, tau=3, max_dim=150)
        fig_r = plot_recurrence_matrix(r_mat, title="Recurrence Plot (Phase Space Trajectory)")
        st.plotly_chart(fig_r, use_container_width=True)

    with c_stat:
        st.subheader("Laboratory Diagnostics")
        st.markdown(f"""
        <div class="qe-card qe-card-accent">
            <h4 style="margin-top:0;">Dataset Configuration</h4>
            <table style="width:100%; font-size:0.88rem; color:#334155;">
                <tr><td style="padding:4px 0;"><b>Active Dataset:</b></td><td>{st.session_state.get('dataset_name', 'None')}</td></tr>
                <tr><td style="padding:4px 0;"><b>Signal Channel:</b></td><td><code>{sig_name}</code></td></tr>
                <tr><td style="padding:4px 0;"><b>Temporal Channel:</b></td><td><code>{st.session_state.get('time_col', 'Row Index')}</code></td></tr>
                <tr><td style="padding:4px 0;"><b>Sampling Frequency:</b></td><td>{f'{fs} Hz' if fs else 'Normalized (Sample Index)'}</td></tr>
                <tr><td style="padding:4px 0;"><b>Applied Filters:</b></td><td>{len(st.session_state.get('preprocessor').pipeline_history) if st.session_state.get('preprocessor') else 0} transformations</td></tr>
                <tr><td style="padding:4px 0;"><b>Recurrence Rate (RR):</b></td><td>{rqa_meta.get('Recurrence Rate (RR)', 0.0):.4f}</td></tr>
                <tr><td style="padding:4px 0;"><b>Determinism (DET):</b></td><td>{rqa_meta.get('Determinism (DET)', 0.0)*100:.1f}%</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 **Navigation Tip**: Use the sidebar to inspect the dataset, apply signal preprocessing filters, run nonlinear chaos dynamics, train ML classifiers, or execute Quantum Kernel SVM on the PennyLane simulator.")

