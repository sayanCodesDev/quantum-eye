"""
QUANTUM EYE — Page 4: Time-Domain Signal Analysis
"""
import streamlit as st
import numpy as np
import pandas as pd
from modules.state_manager import get_current_signal
from modules.signal_analysis import (
    compute_time_domain_metrics, plot_waveform, plot_comparison_waveform,
    plot_rolling_statistics, plot_distribution, plot_boxplot
)

def render():
    st.title("📈 Time-Domain Signal Analysis")
    st.markdown("Examine the waveform dynamics, amplitude statistical moments, rolling dispersion envelopes, and distribution spread.")

    time_axis, signal_values, sig_name, fs = get_current_signal()
    if signal_values is None or len(signal_values) == 0:
        st.warning("No signal loaded. Please select or upload a dataset.")
        return

    prep = st.session_state.get("preprocessor")
    orig_signal = prep.original_df[sig_name].to_numpy(dtype=float) if prep and sig_name in prep.original_df.columns else signal_values

    # 1. Statistical Moments Grid
    metrics = compute_time_domain_metrics(signal_values)
    st.session_state.setdefault("cached_results", {})["time_domain_metrics"] = metrics

    st.subheader(f"📊 Statistical Moments: `{sig_name}`")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mean (μ)", f"{metrics.get('Mean', 0.0):.4f}")
    col2.metric("Std Dev (σ)", f"{metrics.get('Std Dev', 0.0):.4f}")
    col3.metric("Variance (σ²)", f"{metrics.get('Variance', 0.0):.4f}")
    col4.metric("Skewness", f"{metrics.get('Skewness', 0.0):.4f}")
    col5.metric("Excess Kurtosis", f"{metrics.get('Excess Kurtosis', 0.0):.4f}")

    col6, col7, col8, col9, col10 = st.columns(5)
    col6.metric("RMS Amplitude", f"{metrics.get('RMS', 0.0):.4f}")
    col7.metric("Peak-to-Peak", f"{metrics.get('Peak-to-Peak', 0.0):.4f}")
    col8.metric("Crest Factor", f"{metrics.get('Crest Factor', 0.0):.3f}")
    col9.metric("Min / Max", f"{metrics.get('Min', 0.0):.2f} / {metrics.get('Max', 0.0):.2f}")
    col10.metric("Zero Crossings", f"{metrics.get('Zero Crossings', 0):,}")

    st.markdown("---")

    # 2. Interactive Waveform Plots
    t_label = f"Time ({'s' if fs else 'Sample Index'})"
    
    tab_wave, tab_comp, tab_roll, tab_dist = st.tabs([
        "🌊 Processed Waveform", "⚖️ Original vs. Processed", "📉 Rolling Trends & Bands", "📊 Distribution & Box Plot"
    ])

    with tab_wave:
        st.subheader("Signal Waveform")
        fig_w = plot_waveform(
            time_axis=time_axis,
            signal_values=signal_values,
            title=f"Waveform: {sig_name} (Length: {len(signal_values):,} points)",
            signal_name=sig_name,
            x_label=t_label
        )
        st.plotly_chart(fig_w, use_container_width=True)

    with tab_comp:
        st.subheader("Original vs. Processed Comparison")
        if len(orig_signal) == len(signal_values):
            fig_c = plot_comparison_waveform(
                time_axis=time_axis,
                orig_signal=orig_signal,
                proc_signal=signal_values,
                orig_name="Original Raw",
                proc_name="Processed",
                x_label=t_label
            )
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            st.info("Signal lengths differ due to row filtering. Reset pipeline to compare overlays.")

    with tab_roll:
        st.subheader("Rolling Statistics & Uncertainty Envelope")
        c_win, _ = st.columns([1, 2])
        with c_win:
            roll_w = st.slider("Rolling Window Size", min_value=5, max_value=min(1000, max(10, len(signal_values)//2)), value=min(50, len(signal_values)//4), step=5)
        fig_roll = plot_rolling_statistics(
            time_axis=time_axis,
            signal_values=signal_values,
            window_size=roll_w,
            x_label=t_label
        )
        st.plotly_chart(fig_roll, use_container_width=True)

    with tab_dist:
        st.subheader("Amplitude Probability Density & Spread")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            n_bins = st.slider("Histogram Bins", min_value=10, max_value=100, value=40, step=5)
            fig_hist = plot_distribution(signal_values, bins=n_bins)
            st.plotly_chart(fig_hist, use_container_width=True)
        with d_col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            fig_box = plot_boxplot(signal_values, signal_name=sig_name)
            st.plotly_chart(fig_box, use_container_width=True)

