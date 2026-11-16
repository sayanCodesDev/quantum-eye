"""
QUANTUM EYE — Page 2: Upload Dataset & Inspection
"""
import streamlit as st
import pandas as pd
import numpy as np
from modules.data_loader import load_dataset, inspect_dataset
from modules.state_manager import set_active_dataset

def render():
    st.title("📂 Dataset Upload & Laboratory Ingestion")
    st.markdown("Upload almost any numerical dataset or select a pre-loaded scientific benchmark signal to begin intelligent analysis.")

    tab_upload, tab_benchmarks = st.tabs(["📤 Upload Custom Dataset", "🧪 Pre-loaded Scientific Benchmarks"])

    # --- TAB 1: CUSTOM FILE UPLOAD ---
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose a CSV, Excel (.xlsx, .xls), or JSON dataset",
            type=["csv", "xlsx", "xls", "json"],
            help="Supports standard scientific datasets up to 100MB."
        )

        if uploaded_file is not None:
            df, err = load_dataset(uploaded_file, file_name=uploaded_file.name)
            if err:
                st.error(err)
            else:
                st.success(f"Successfully loaded file `{uploaded_file.name}` ({len(df):,} rows, {len(df.columns)} columns).")
                if st.button("Ingest Uploaded Dataset", type="primary"):
                    set_active_dataset(df, name=uploaded_file.name)
                    st.rerun()

    # --- TAB 2: BENCHMARK DATASETS ---
    with tab_benchmarks:
        st.markdown("""
        Select from standardized time-series benchmarks inspired by the **UCR/UEA Time Series Classification Archive**
        ([timeseriesclassification.com](https://www.timeseriesclassification.com/)):
        """)
        
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("🫀 ECG200 Heartbeat", use_container_width=True):
                df_b, _ = load_dataset("data/ecg200_sample.csv")
                set_active_dataset(df_b, "ECG200 Heartbeat", signal_col="ecg_lead_ii", time_col="timestamp", fs=128.0, group_col="diagnostic_label")
                st.success("Loaded ECG200 Heartbeat benchmark!")
                st.rerun()
            st.caption("Electrocardiogram signals labeled Normal vs. Myocardial Infarction.")

        with b2:
            if st.button("📊 Synthetic Control", use_container_width=True):
                df_b, _ = load_dataset("data/synthetic_control.csv")
                set_active_dataset(df_b, "Synthetic Control Charts", signal_col="sensor_value", time_col="time_step", fs=1.0, group_col="process_class")
                st.success("Loaded Synthetic Control benchmark!")
                st.rerun()
            st.caption("6 industrial process control modes: Cyclic, Shift, Trend, Normal.")

        with b3:
            if st.button("🌀 Lorenz Chaos", use_container_width=True):
                df_b, _ = load_dataset("data/lorenz_chaos.csv")
                set_active_dataset(df_b, "Lorenz Attractor (Chaos)", signal_col="lorenz_x", time_col="time", fs=100.0)
                st.success("Loaded Lorenz Attractor benchmark!")
                st.rerun()
            st.caption("Deterministic chaotic system coordinates (σ=10, ρ=28, β=8/3).")

        with b4:
            if st.button("⏳ Mackey-Glass Chaos", use_container_width=True):
                df_b, _ = load_dataset("data/mackey_glass.csv")
                set_active_dataset(df_b, "Mackey-Glass (Delayed DE)", signal_col="mackey_glass_signal", time_col="time_sec", fs=10.0)
                st.success("Loaded Mackey-Glass benchmark!")
                st.rerun()
            st.caption("Nonlinear delay-differential equation with high-dimensional attractor.")

    st.markdown("---")

    # --- INSPECTION OF CURRENT DATASET ---
    df_raw = st.session_state.get("df_raw", None)
    if df_raw is None:
        st.info("No dataset is currently loaded in memory. Upload a file above or pick a benchmark.")
        return

    inspection = st.session_state.get("inspection", inspect_dataset(df_raw, st.session_state.get("dataset_name", "Dataset")))
    
    st.subheader(f"🔍 Automated Inspection: {inspection['dataset_name']}")

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows", f"{inspection['n_rows']:,}")
    m2.metric("Total Columns", f"{inspection['n_cols']}")
    m3.metric("Missing Values", f"{inspection['total_missing']:,} ({inspection['missing_pct']}%)")
    m4.metric("Duplicate Rows", f"{inspection['duplicate_rows']:,} ({inspection['duplicate_pct']}%)")

    # Sequentiality assessment alert
    is_seq = inspection["is_sequential"]
    seq_reason = inspection["seq_reason"]
    if is_seq:
        st.markdown(f"""
        <div class="qe-callout qe-callout-success">
            <b>Sequential Signal Detected:</b> {seq_reason}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="qe-callout qe-callout-warning">
            <b>Static / Tabular Data Notice:</b> {seq_reason}<br>
            <i>Quantum Eye will support both temporal signal processing and standard static distribution analysis.</i>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Signal Configuration
    st.subheader("⚙️ Signal Configuration & Channel Mapping")
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)

    num_cols = inspection["num_cols"]
    all_cols = inspection["columns"]
    time_candidates = inspection["timestamp_candidates"]

    with col_c1:
        current_sig = st.session_state.get("signal_col", num_cols[0] if num_cols else None)
        sig_idx = num_cols.index(current_sig) if current_sig in num_cols else 0
        new_sig = st.selectbox(
            "Target Signal Column",
            options=num_cols if num_cols else ["No numeric columns"],
            index=sig_idx,
            help="Select the continuous numerical signal to analyze."
        )

    with col_c2:
        time_options = ["Row Index"] + all_cols
        current_time = st.session_state.get("time_col", "Row Index")
        time_idx = time_options.index(current_time) if current_time in time_options else 0
        new_time = st.selectbox(
            "Temporal / Time Axis Column",
            options=time_options,
            index=time_idx,
            help="Select timestamp column, or use Row Index if no temporal axis exists."
        )

    with col_c3:
        current_fs = st.session_state.get("fs", 100.0) or 100.0
        known_fs = st.checkbox("Known Sampling Rate (Hz)?", value=(st.session_state.get("fs") is not None))
        if known_fs:
            new_fs = st.number_input("Sampling Frequency (Hz)", value=float(current_fs), min_value=0.01, step=10.0)
        else:
            new_fs = None

    with col_c4:
        cat_options = ["None"] + all_cols
        current_grp = st.session_state.get("group_col", "None")
        grp_idx = cat_options.index(current_grp) if current_grp in cat_options else 0
        new_grp = st.selectbox(
            "Group / Label Column (Optional)",
            options=cat_options,
            index=grp_idx,
            help="For supervised classification or stratified sub-group analysis."
        )

    if st.button("Apply Signal Configuration", type="primary"):
        st.session_state["signal_col"] = new_sig
        st.session_state["time_col"] = new_time
        st.session_state["fs"] = new_fs
        st.session_state["group_col"] = None if new_grp == "None" else new_grp
        st.success("Signal configuration updated!")
        st.rerun()

    # Data Types & Statistics Explorer
    st.subheader("📋 Column Profiles & Summary Statistics")
    col_t1, col_t2 = st.columns([1, 1.4])
    with col_t1:
        st.markdown("**Column Schema**")
        schema_df = pd.DataFrame({
            "Column": inspection["columns"],
            "Data Type": [inspection["dtypes"][c] for c in inspection["columns"]],
            "Type Category": ["Numerical" if c in num_cols else "Categorical/Object" for c in inspection["columns"]],
            "Missing": [inspection["missing_counts"][c] for c in inspection["columns"]]
        })
        st.dataframe(schema_df, use_container_width=True, height=260)

    with col_t2:
        st.markdown("**Numerical Summary Statistics**")
        if not inspection["summary_stats"].empty:
            st.dataframe(inspection["summary_stats"].round(4), use_container_width=True, height=260)
        else:
            st.info("No numerical summary statistics available.")

