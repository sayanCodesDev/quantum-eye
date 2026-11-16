"""
QUANTUM EYE — Page 3: Dataset Explorer & Interactive Preprocessing
"""
import streamlit as st
import pandas as pd
import numpy as np
from modules.state_manager import get_current_signal
from modules.signal_analysis import plot_comparison_waveform

def render():
    st.title("🔬 Dataset Explorer & Signal Preprocessing")
    st.markdown("Inspect raw data records and apply scientific filters, smoothing, normalization, and outlier corrections. Original data remains pristine.")

    prep = st.session_state.get("preprocessor", None)
    if prep is None:
        st.warning("No dataset loaded. Please visit 'Upload Dataset' first.")
        return

    sig_col = st.session_state.get("signal_col", None)
    fs = st.session_state.get("fs", 100.0) or 100.0

    # Tabs: Data Explorer vs Preprocessing Controls
    tab_view, tab_prep, tab_history = st.tabs(["📊 Data Table & Distribution", "🛠️ Preprocessing Operations", "📜 Pipeline History"])

    with tab_view:
        st.subheader("Interactive Data View")
        c1, c2 = st.columns([1, 3])
        with c1:
            view_mode = st.radio("Display Version", ["Processed Data", "Original Raw Data"], horizontal=True)
            n_rows_show = st.slider("Display Rows", min_value=10, max_value=500, value=50, step=10)
        
        target_df = prep.processed_df if view_mode == "Processed Data" else prep.original_df
        st.dataframe(target_df.head(n_rows_show), use_container_width=True)
        st.caption(f"Showing first {min(n_rows_show, len(target_df))} of {len(target_df):,} rows.")

    with tab_prep:
        st.subheader(f"Apply Transformations on Active Signal: `{sig_col}`")
        if not sig_col:
            st.error("No numerical signal column selected.")
            return

        op_category = st.selectbox(
            "Select Preprocessing Operation",
            [
                "Missing Value Imputation",
                "Duplicate Removal",
                "Outlier Detection & Handling",
                "Scaling & Normalization",
                "Moving-Average Smoothing",
                "Median Filtering",
                "Savitzky-Golay Polynomial Filter",
                "Butterworth Filter (Digital)",
                "Detrending (Linear / Constant)"
            ]
        )

        st.markdown("---")

        if op_category == "Missing Value Imputation":
            st.markdown("**Missing Value Imputation**")
            miss_count = int(prep.processed_df[sig_col].isnull().sum())
            st.write(f"Current missing values in `{sig_col}`: **{miss_count}**")
            method = st.selectbox("Imputation Technique", ["interpolate", "mean", "median", "drop", "ffill", "bfill"],
                                  format_func=lambda x: {
                                      "interpolate": "Linear Interpolation (Continuous)",
                                      "mean": "Mean Value Imputation",
                                      "median": "Median Value Imputation (Robust)",
                                      "drop": "Drop Incomplete Rows",
                                      "ffill": "Forward Fill (Carry Last Observation)",
                                      "bfill": "Backward Fill (Next Observation)"
                                  }.get(x, x))
            if st.button("Apply Missing Value Imputation", type="primary"):
                msg = prep.handle_missing(sig_col, method=method)
                st.success(msg)
                st.rerun()

        elif op_category == "Duplicate Removal":
            st.markdown("**Duplicate Row Removal**")
            n_dups = int(prep.processed_df.duplicated().sum())
            st.write(f"Total duplicate rows detected: **{n_dups}**")
            if st.button("Purge Duplicates", type="primary", disabled=(n_dups == 0)):
                msg = prep.remove_duplicates()
                st.success(msg)
                st.rerun()

        elif op_category == "Outlier Detection & Handling":
            st.markdown("**Statistical Outlier Handling**")
            out_method = st.radio("Detection Criterion", ["iqr", "zscore"], format_func=lambda x: "Interquartile Range (IQR)" if x == "iqr" else "Z-Score (|z| > threshold)", horizontal=True)
            c_thresh, c_act = st.columns(2)
            with c_thresh:
                thresh_val = st.number_input("Threshold", value=1.5 if out_method == "iqr" else 3.0, step=0.5)
            with c_act:
                act = st.selectbox("Action", ["clip", "remove"], format_func=lambda x: "Clip / Winsorize (Cap Bounds)" if x == "clip" else "Remove Outlier Rows")

            _, meta = prep.detect_outliers(sig_col, method=out_method, threshold=thresh_val)
            st.info(f"Detected **{meta.get('count', 0)} outliers** ({meta.get('percentage', 0.0)}% of dataset) under current threshold.")

            if st.button("Execute Outlier Correction", type="primary"):
                msg = prep.handle_outliers(sig_col, method=out_method, action=act, threshold=thresh_val)
                st.success(msg)
                st.rerun()

        elif op_category == "Scaling & Normalization":
            st.markdown("**Feature Scaling & Amplitude Normalization**")
            scale_mode = st.selectbox("Scaling Method", ["standard", "minmax", "robust"],
                                      format_func=lambda x: {
                                          "standard": "Z-Score Standardization (μ = 0, σ = 1)",
                                          "minmax": "Min-Max Normalization ([0, 1] range)",
                                          "robust": "Robust Scaling (Median & Interquartile Range)"
                                      }.get(x, x))
            if st.button("Apply Scaling", type="primary"):
                msg = prep.scale_signal(sig_col, method=scale_mode)
                st.success(msg)
                st.rerun()

        elif op_category == "Moving-Average Smoothing":
            st.markdown("**Rolling Moving-Average Smoothing**")
            window_size = st.slider("Window Size (samples)", min_value=3, max_value=101, value=7, step=2)
            if st.button("Apply Moving Average", type="primary"):
                msg = prep.apply_moving_average(sig_col, window=window_size)
                st.success(msg)
                st.rerun()

        elif op_category == "Median Filtering":
            st.markdown("**Nonlinear Median Filter (Impulsive Noise Suppression)**")
            kernel_s = st.slider("Kernel Size (odd integer)", min_value=3, max_value=31, value=5, step=2)
            if st.button("Apply Median Filter", type="primary"):
                msg = prep.apply_median_filter(sig_col, kernel_size=kernel_s)
                st.success(msg)
                st.rerun()

        elif op_category == "Savitzky-Golay Polynomial Filter":
            st.markdown("**Savitzky-Golay Polynomial Smoothing** (Preserves signal peaks while removing high-frequency noise)")
            c_sg1, c_sg2 = st.columns(2)
            with c_sg1:
                sg_win = st.slider("Window Length (odd)", min_value=5, max_value=51, value=11, step=2)
            with c_sg2:
                sg_poly = st.slider("Polynomial Order", min_value=1, max_value=5, value=3)
            if st.button("Apply Savitzky-Golay", type="primary"):
                msg = prep.apply_savgol_filter(sig_col, window_length=sg_win, polyorder=sg_poly)
                st.success(msg)
                st.rerun()

        elif op_category == "Butterworth Filter (Digital)":
            st.markdown("**Butterworth Digital Filter (IIR Zero-Phase Filtering)**")
            btype = st.selectbox("Filter Topology", ["low", "high", "band"], format_func=lambda x: {
                "low": "Low-Pass (Remove high-frequency noise)",
                "high": "High-Pass (Remove low-frequency baseline drift)",
                "band": "Band-Pass (Isolate frequency band)"
            }.get(x, x))

            nyq = 0.5 * fs
            st.caption(f"Nyquist frequency = {nyq:.1f} Hz (based on sampling rate fs = {fs:.1f} Hz).")

            c_bf1, c_bf2 = st.columns(2)
            with c_bf1:
                b_order = st.slider("Filter Order", min_value=1, max_value=8, value=4)
            with c_bf2:
                if btype in ["low", "high"]:
                    b_cut = st.number_input(f"Cutoff Frequency (Hz, < {nyq:.1f} Hz)", value=min(15.0, nyq * 0.4), min_value=0.1, max_value=nyq * 0.99)
                else:
                    low_f = st.number_input("Low Cutoff (Hz)", value=1.0, min_value=0.1, max_value=nyq * 0.5)
                    high_f = st.number_input("High Cutoff (Hz)", value=min(20.0, nyq * 0.8), min_value=low_f + 0.1, max_value=nyq * 0.99)
                    b_cut = (low_f, high_f)

            if st.button("Apply Butterworth Filter", type="primary"):
                msg = prep.apply_butterworth_filter(sig_col, btype=btype, cutoff=b_cut, fs=fs, order=b_order)
                if "failed" in msg or "invalid" in msg.lower():
                    st.error(msg)
                else:
                    st.success(msg)
                    st.rerun()

        elif op_category == "Detrending (Linear / Constant)":
            st.markdown("**Signal Detrending**")
            d_type = st.radio("Detrend Mode", ["linear", "constant"], format_func=lambda x: "Linear Trend Removal" if x == "linear" else "Constant Mean Removal", horizontal=True)
            if st.button("Apply Detrending", type="primary"):
                msg = prep.apply_detrend(sig_col, detrend_type=d_type)
                st.success(msg)
                st.rerun()

    with tab_history:
        st.subheader("Transformation Pipeline History")
        history = prep.pipeline_history
        if not history:
            st.info("No transformations have been applied yet. The dataset is in its initial state.")
        else:
            hist_df = pd.DataFrame(history)
            st.dataframe(hist_df, use_container_width=True)
            if st.button("🔄 Reset to Original Raw Data", type="secondary"):
                prep.reset()
                st.success("Dataset successfully reset to raw state!")
                st.rerun()

    # Before-vs-After Comparison Waveform
    st.markdown("---")
    st.subheader("🔍 Real-Time Waveform Comparison: Original vs. Processed")
    time_axis, proc_signal, _, _ = get_current_signal()
    orig_signal = prep.original_df[sig_col].to_numpy(dtype=float)

    if proc_signal is not None and len(proc_signal) > 0 and len(orig_signal) == len(proc_signal):
        fig_comp = plot_comparison_waveform(
            time_axis=time_axis,
            orig_signal=orig_signal,
            proc_signal=proc_signal,
            orig_name="Original Raw Signal",
            proc_name="Processed Signal",
            x_label=f"Time ({'s' if st.session_state.get('fs') else 'Sample'})"
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Row dimensions differed due to row filtering. Reset pipeline to compare overlays.")

