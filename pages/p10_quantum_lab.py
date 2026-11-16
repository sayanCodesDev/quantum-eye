"""
QUANTUM EYE — Page 10: Quantum Machine Learning Laboratory
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from modules.quantum_engine import QuantumEngine, PENNYLANE_AVAILABLE
from modules.ml_engine import MLEngine

def render():
    st.title("⚛️ Quantum Machine Learning Laboratory")
    st.markdown("Experimental quantum feature mapping via Hilbert space projection. Explore Quantum Kernel Support Vector Machines (QSVM) on PennyLane quantum state-vector simulators.")

    # Scientific Transparency Note
    st.markdown("""
    <div class="qe-callout qe-callout-teal">
        <b>Quantum Computing Research Principle:</b> Quantum computing is an exploratory computational paradigm.
        Quantum feature maps project classical data into high-dimensional Hilbert spaces to discover nonlinear hyperplanes;
        however, state-vector simulation on classical CPUs incurs simulation overhead and is <b>not</b> inherently faster than classical kernels.
        Quantum Eye reports exact execution metrics without fabricated speedup claims.
    </div>
    """, unsafe_allow_html=True)

    if not PENNYLANE_AVAILABLE:
        st.error("PennyLane library is not available in the current environment.")
        return

    # Backend Telemetry Configuration
    st.subheader("🖥️ Quantum Backend Configuration")
    c_b1, c_b2, c_b3 = st.columns(3)
    with c_b1:
        backend_choice = st.selectbox("Quantum Simulator Backend", ["default.qubit", "lightning.qubit"],
                                      help="PennyLane state-vector simulator device.")
    with c_b2:
        n_qubits = st.slider("Qubit Allocation (PCA Dimension)", min_value=2, max_value=4, value=2,
                             help="Controls feature dimensionality and Hilbert space dimension (2^n states).")
    with c_b3:
        svm_c = st.number_input("SVM Regularization (C)", value=1.0, min_value=0.1, step=0.5)

    st.markdown("---")

    # Workflow Setup
    st.subheader("🧪 Quantum Kernel Classification Workflow")
    st.markdown("""
    1. **Data Ingestion**: Takes independent signals or active dataset features.
    2. **Dimensionality Reduction**: Fits PCA to reduce feature subspace to $N_{qubits}$ orthogonal components.
    3. **Angle & Entanglement Encoding**: Applies **ZZ Quantum Feature Map** ($H \\to R_z(2x_i) \\to \\text{CNOT} \\to R_z(2(\\pi-x_i)(\\pi-x_j)) \\to \\text{CNOT}$).
    4. **Quantum Kernel Gram Matrix**: Measures quantum state fidelity $K_{ij} = |\\langle\\phi(x_i)|\\phi(x_j)\\rangle|^2$.
    5. **Empirical Benchmarking**: Compares Quantum Kernel SVM against Classical RBF SVM baseline.
    """)

    # Data Source Selection
    ds_source = st.radio("Dataset for Quantum Experiment", ["Controlled 3-Class Signals (Periodic vs Stochastic vs Chaotic)", "Active Loaded Dataset"], horizontal=True)

    if ds_source == "Controlled 3-Class Signals (Periodic vs Stochastic vs Chaotic)":
        c_q1, c_q2 = st.columns(2)
        with c_q1:
            q_samples_per_class = st.slider("Samples per Class (Keep small for exact QPU simulation)", min_value=6, max_value=20, value=8, step=2)
        with c_q2:
            q_sig_len = st.slider("Signal Sample Length", min_value=150, max_value=500, value=250, step=50)

    if st.button("⚡ Execute Quantum Kernel SVM Experiment", type="primary"):
        with st.spinner(f"Simulating {n_qubits}-qubit quantum circuits on {backend_choice}..."):
            try:
                if ds_source.startswith("Controlled"):
                    X_df, y_ser = MLEngine.run_signal_intelligence_experiment(
                        n_signals_per_class=q_samples_per_class,
                        signal_length=q_sig_len,
                        noise_snr_db=20.0
                    )
                    X_data = X_df.values
                    y_data = y_ser.values
                else:
                    prep = st.session_state.get("preprocessor")
                    num_cols = list(prep.processed_df.select_dtypes(include=[np.number]).columns)
                    target_c = st.session_state.get("group_col") or num_cols[-1]
                    sub_df = prep.processed_df.dropna(subset=num_cols).head(30)
                    X_data = sub_df[[c for c in num_cols if c != target_c]].values
                    y_data = sub_df[target_c].values

                # Train-test split (60% train, 40% test)
                split_idx = int(len(X_data) * 0.65)
                X_tr, X_te = X_data[:split_idx], X_data[split_idx:]
                y_tr, y_te = y_data[:split_idx], y_data[split_idx:]

                qe = QuantumEngine(n_qubits=n_qubits, backend_name=backend_choice)
                q_results = qe.train_quantum_kernel_svm(X_tr, y_tr, X_te, y_te, C=svm_c)
                st.session_state["quantum_results"] = q_results

            except Exception as e:
                st.error(f"Quantum simulation error: {str(e)}")

    # Display Quantum Results
    if "quantum_results" in st.session_state and st.session_state["quantum_results"] is not None:
        q_res = st.session_state["quantum_results"]
        st.markdown("---")
        st.subheader("📊 Quantum vs. Classical Baseline Results")

        q_acc = q_res["quantum"]["accuracy"]
        c_acc = q_res["classical_baseline"]["accuracy"]
        q_f1 = q_res["quantum"]["f1_score"]
        c_f1 = q_res["classical_baseline"]["f1_score"]
        q_time = q_res["quantum"]["total_time"]
        c_time = q_res["classical_baseline"]["train_time"] + q_res["classical_baseline"]["inference_time"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Quantum Kernel Accuracy", f"{q_acc*100:.1f}%", delta=f"{(q_acc - c_acc)*100:+.1f}% vs Classical")
        m2.metric("Quantum F1-Score", f"{q_f1:.4f}", delta=f"{(q_f1 - c_f1):+.4f}")
        m3.metric("Quantum Simulation Time", f"{q_time:.3f} s", delta=f"{q_res['telemetry']['quantum_overhead_ratio']:.1f}x Overhead", delta_color="inverse")
        m4.metric("Classical Baseline Time", f"{c_time:.4f} s")

        # Telemetry & Quantum Kernel Heatmap
        col_km, col_telem = st.columns([1.2, 1])
        with col_km:
            st.markdown("**Quantum Gram Kernel Matrix ($K_{train}$)**")
            K_tr = q_res["quantum"]["K_train"]
            fig_km = go.Figure(data=go.Heatmap(
                z=K_tr,
                colorscale='Viridis',
                zmin=0.0,
                zmax=1.0,
                colorbar=dict(title="Fidelity |⟨ϕ_i|ϕ_j⟩|²")
            ))
            fig_km.update_layout(
                title=f"Quantum Kernel Gram Matrix ({K_tr.shape[0]}x{K_tr.shape[1]})",
                xaxis=dict(title="Sample Index i", gridcolor="#f1f5f9"),
                yaxis=dict(title="Sample Index j", gridcolor="#f1f5f9"),
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                height=350,
                margin=dict(l=40, r=20, t=40, b=40)
            )
            st.plotly_chart(fig_km, use_container_width=True)

        with col_telem:
            st.markdown("**Circuit & Telemetry Diagnostics**")
            telem = q_res["quantum"]["train_telemetry"]
            st.markdown(f"""
            <div class="qe-card qe-card-violet">
                <table style="width:100%; font-size:0.88rem; color:#334155;">
                    <tr><td style="padding:4px 0;"><b>Backend Architecture:</b></td><td><code>{q_res['telemetry']['backend']}</code></td></tr>
                    <tr><td style="padding:4px 0;"><b>Physical Hardware Status:</b></td><td>Local State-Vector Simulator</td></tr>
                    <tr><td style="padding:4px 0;"><b>Allocated Qubits:</b></td><td>{q_res['telemetry']['n_qubits']} qubits (dim = {2**q_res['telemetry']['n_qubits']})</td></tr>
                    <tr><td style="padding:4px 0;"><b>Estimated Circuit Depth:</b></td><td>{telem['circuit_depth']} gates</td></tr>
                    <tr><td style="padding:4px 0;"><b>Total Circuit Executions:</b></td><td>{telem['circuit_evaluations'] + q_res['quantum']['test_telemetry']['circuit_evaluations']} circuits</td></tr>
                    <tr><td style="padding:4px 0;"><b>Execution Latency:</b></td><td>{telem['time_per_evaluation_ms']:.2f} ms / evaluation</td></tr>
                    <tr><td style="padding:4px 0;"><b>Variance Captured by PCA:</b></td><td>{q_res['telemetry']['total_pca_variance']*100:.1f}%</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Quantum kernel Gram matrix computed via adjoint state overlap.")

