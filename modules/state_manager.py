"""
QUANTUM EYE — Session State Manager
Coordinates shared state across all 12 navigation pages.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from modules.data_loader import load_dataset, inspect_dataset
from modules.preprocessing import Preprocessor

def init_session_state():
    """Initializes all shared session state variables if not already present."""
    if "initialized" not in st.session_state:
        # Load default benchmark dataset (ECG200 Heartbeat)
        df_default, _ = load_dataset("data/ecg200_sample.csv")
        if df_default is not None:
            st.session_state["df_raw"] = df_default.copy()
            st.session_state["preprocessor"] = Preprocessor(df_default)
            st.session_state["dataset_name"] = "ECG200 Heartbeat (Benchmark)"
            st.session_state["signal_col"] = "ecg_lead_ii"
            st.session_state["time_col"] = "timestamp"
            st.session_state["fs"] = 128.0
            st.session_state["group_col"] = "diagnostic_label"
            st.session_state["inspection"] = inspect_dataset(df_default, "ECG200 Heartbeat (Benchmark)")
        else:
            st.session_state["df_raw"] = None
            st.session_state["preprocessor"] = None
            st.session_state["dataset_name"] = "No Dataset Loaded"
            st.session_state["signal_col"] = None
            st.session_state["time_col"] = None
            st.session_state["fs"] = None
            st.session_state["group_col"] = None
            st.session_state["inspection"] = None

        st.session_state["cached_results"] = {}
        st.session_state["ml_results"] = None
        st.session_state["quantum_results"] = None
        st.session_state["anomaly_results"] = None
        st.session_state["api_key"] = ""
        st.session_state["llm_provider"] = "Gemini"
        st.session_state["initialized"] = True


def set_active_dataset(df: pd.DataFrame, name: str, signal_col: Optional[str] = None, time_col: Optional[str] = None, fs: Optional[float] = None, group_col: Optional[str] = None):
    """Sets new active dataset and reinitializes preprocessor and inspection."""
    st.session_state["df_raw"] = df.copy()
    st.session_state["preprocessor"] = Preprocessor(df)
    st.session_state["dataset_name"] = name
    inspection = inspect_dataset(df, name)
    st.session_state["inspection"] = inspection

    # Auto-select signal column
    num_cols = inspection["num_cols"]
    candidates = inspection["timestamp_candidates"]

    # Time column
    if time_col and time_col in df.columns:
        st.session_state["time_col"] = time_col
    elif candidates:
        st.session_state["time_col"] = candidates[0]
    else:
        st.session_state["time_col"] = "Row Index"

    # Signal column
    eligible_signals = [c for c in num_cols if c != st.session_state["time_col"]]
    if signal_col and signal_col in df.columns:
        st.session_state["signal_col"] = signal_col
    elif eligible_signals:
        st.session_state["signal_col"] = eligible_signals[0]
    elif num_cols:
        st.session_state["signal_col"] = num_cols[0]
    else:
        st.session_state["signal_col"] = None

    st.session_state["fs"] = fs
    st.session_state["group_col"] = group_col
    st.session_state["cached_results"] = {}
    st.session_state["ml_results"] = None
    st.session_state["quantum_results"] = None
    st.session_state["anomaly_results"] = None


def get_current_signal() -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, Optional[float]]:
    """
    Returns (time_axis, signal_values, signal_col_name, fs).
    Signal values come from the processed dataframe to reflect active preprocessing pipeline.
    """
    if "preprocessor" not in st.session_state or st.session_state["preprocessor"] is None:
        return None, None, "", None

    df_proc = st.session_state["preprocessor"].processed_df
    sig_col = st.session_state.get("signal_col", None)
    time_col = st.session_state.get("time_col", None)
    fs = st.session_state.get("fs", None)

    if not sig_col or sig_col not in df_proc.columns:
        return None, None, "", None

    signal_values = df_proc[sig_col].to_numpy(dtype=float)
    n = len(signal_values)

    if time_col and time_col in df_proc.columns and time_col != "Row Index":
        t_vals = pd.to_numeric(df_proc[time_col], errors='coerce').fillna(0).to_numpy()
        # Verify if time is monotonic and spaced
        if len(t_vals) > 1 and np.all(np.diff(t_vals) > 0) and fs is None:
            # estimate fs from dt
            median_dt = np.median(np.diff(t_vals))
            if median_dt > 0:
                fs = round(1.0 / median_dt, 2)
        time_axis = t_vals
    elif fs is not None and fs > 0:
        time_axis = np.arange(n) / fs
    else:
        time_axis = np.arange(n, dtype=float)

    return time_axis, signal_values, sig_col, fs

