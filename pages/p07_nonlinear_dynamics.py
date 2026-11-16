"""
QUANTUM EYE — Page 7: Nonlinear Dynamics & Chaos Laboratory
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from modules.state_manager import get_current_signal
from modules.nonlinear_analysis import (
    compute_shannon_entropy, compute_permutation_entropy, compute_approximate_entropy,
    compute_sample_entropy, reconstruct_phase_space, plot_phase_space,
    estimate_lyapunov_exponent, compute_recurrence_matrix, plot_recurrence_matrix
)

def render():
    st.title("🌀 Nonlinear Dynamics & Chaos Laboratory")
    st.markdown("Investigate phase space reconstruction, deterministic chaos, trajectory divergence, and recurrence topology in complex systems.")

    # Scientific Caution Callout
    st.markdown("""
    <div class="qe-callout qe-callout-warning">
        <b>Scientific Precaution:</b> Mathematical measures like the Lyapunov exponent or entropy do not in isolation
        conclusively prove that an empirical signal is deterministic chaos or stochastic noise. Finite sample lengths,
        observational noise, and non-stationarity require rigorous joint validation across multiple analytical perspectives.
    </div>
    """, unsafe_allow_html=True)

    time_axis, signal_values, sig_name, fs = get_current_signal()
    if signal_values is None or len(signal_values) == 0:
        st.warning("No signal loaded. Please select or upload a dataset.")
        return

    # Embedding & Delay Controls
    st.subheader("⚙️ Phase Space Reconstruction Parameters (Takens' Theorem)")
    c_m, c_tau, c_win = st.columns(3)
    with c_m:
        emb_m = st.slider("Embedding Dimension (m)", min_value=2, max_value=6, value=3, help="Number of orthogonal delay coordinates.")
    with c_tau:
        emb_tau = st.slider("Time Delay (τ samples)", min_value=1, max_value=25, value=3, help="Temporal spacing between embedding coordinates.")
    with c_win:
        max_samples_nl = st.slider("Analysis Window Size", min_value=200, max_value=min(5000, len(signal_values)), value=min(1500, len(signal_values)), step=100)

    # Subsample for nonlinear analysis window
    sig_sub = signal_values[:max_samples_nl]

    tab_entropy, tab_phase, tab_lyap, tab_rqa = st.tabs([
        "📊 Entropy Complexity Comparison", "🪐 Phase Space Attractor", "📈 Lyapunov Divergence (λ₁)", "🔁 Recurrence Plot & RQA"
    ])

    # 1. Entropy Comparison
    with tab_entropy:
        st.subheader("Multiscale Entropy Measurements")
        with st.spinner("Calculating entropy metrics..."):
            shannon_val = compute_shannon_entropy(sig_sub)
            pe_val, norm_pe = compute_permutation_entropy(sig_sub, m=emb_m, tau=emb_tau)
            # Use smaller slice for ApEn/SampEn to keep responsive
            apen_val = compute_approximate_entropy(sig_sub[:min(len(sig_sub), 800)], m=2)
            sampen_val = compute_sample_entropy(sig_sub[:min(len(sig_sub), 800)], m=2)

        st.session_state.setdefault("cached_results", {}).setdefault("nonlinear_metrics", {})["shannon_entropy"] = shannon_val
        st.session_state["cached_results"]["nonlinear_metrics"]["perm_entropy"] = norm_pe
        st.session_state["cached_results"]["nonlinear_metrics"]["approx_entropy"] = apen_val
        st.session_state["cached_results"]["nonlinear_metrics"]["sample_entropy"] = sampen_val

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Shannon Entropy", f"{shannon_val:.4f} bits", help="Measures amplitude distribution uncertainty.")
        e2.metric("Permutation Entropy (Norm)", f"{norm_pe:.4f}", help="Ordinal ranking pattern complexity [0, 1].")
        e3.metric("Approximate Entropy (ApEn)", f"{apen_val:.4f}", help="Sub-sequence predictability (Pincus 1991).")
        e4.metric("Sample Entropy (SampEn)", f"{sampen_val:.4f}", help="Bias-free regularity metric (Richman & Moorman 2000).")

        # Comparative Bar Chart
        ent_df = pd.DataFrame({
            "Metric": ["Normalized Permutation Entropy", "Approximate Entropy (ApEn)", "Sample Entropy (SampEn)", "Shannon / 10"],
            "Value": [norm_pe, apen_val, sampen_val, shannon_val / 10.0]
        })
        fig_ent = go.Figure(go.Bar(
            x=ent_df["Metric"],
            y=ent_df["Value"],
            marker=dict(color=['#0284c7', '#0d9488', '#8b5cf6', '#f59e0b'])
        ))
        fig_ent.update_layout(
            title="Comparison of Entropy Complexity Profiles",
            yaxis=dict(title="Score", gridcolor="#f1f5f9"),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            height=340
        )
        st.plotly_chart(fig_ent, use_container_width=True)

    # 2. Phase Space / State Space Reconstruction
    with tab_phase:
        st.subheader("State-Space Attractor Trajectory")
        embedded = reconstruct_phase_space(sig_sub, m=emb_m, tau=emb_tau)
        if len(embedded) > 0:
            fig_ps = plot_phase_space(embedded, m=emb_m, title=f"Reconstructed Attractor: [x(t), x(t+{emb_tau}), x(t+{2*emb_tau})]")
            st.plotly_chart(fig_ps, use_container_width=True)
            st.caption(f"Reconstructed from 1D scalar series into {emb_m}-dimensional delay coordinate space (Embedding Theorem).")
        else:
            st.error("Insufficient samples for selected delay parameters.")

    # 3. Largest Lyapunov Exponent (LLE)
    with tab_lyap:
        st.subheader("Largest Lyapunov Exponent Estimation (Rosenstein Algorithm)")
        with st.spinner("Computing phase-space neighbor divergence..."):
            lle, t_steps, div_curve = estimate_lyapunov_exponent(sig_sub, m=emb_m, tau=emb_tau, max_t=25)

        st.session_state.setdefault("cached_results", {}).setdefault("nonlinear_metrics", {})["lyapunov_exponent"] = lle

        col_l1, col_l2 = st.columns([1, 2.5])
        with col_l1:
            st.metric("Estimated λ₁", f"{lle:.4f}", delta="Chaotic" if lle > 0.05 else ("Periodic" if abs(lle) <= 0.05 else "Convergent"))
            st.markdown(f"""
            **Interpretation:**
            - **λ₁ > 0 ({lle:.4f})**: Sensitive dependence on initial conditions. Neighboring trajectories diverge exponentially.
            - **λ₁ ≈ 0**: Neutral stability (limit cycle / periodic).
            - **λ₁ < 0**: Trajectories converge toward a stable fixed-point.
            """)

        with col_l2:
            if len(div_curve) > 0:
                fig_div = go.Figure()
                fig_div.add_trace(go.Scatter(
                    x=t_steps,
                    y=div_curve,
                    mode='lines+markers',
                    name='Mean Log Divergence',
                    line=dict(color='#ef4444', width=2)
                ))
                fig_div.update_layout(
                    title="Average Divergence of Trajectories <ln(d(t))>",
                    xaxis=dict(title="Divergence Time Steps (t)", gridcolor="#f1f5f9"),
                    yaxis=dict(title="Mean ln(Divergence)", gridcolor="#f1f5f9"),
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                    height=340
                )
                st.plotly_chart(fig_div, use_container_width=True)

    # 4. Recurrence Plot & RQA
    with tab_rqa:
        st.subheader("Recurrence Plot & Recurrence Quantification Analysis (RQA)")
        c_r1, c_r2 = st.columns([1.2, 1])
        with c_r1:
            r_mat, rqa_meta = compute_recurrence_matrix(sig_sub, m=emb_m, tau=emb_tau, max_dim=200)
            st.session_state.setdefault("cached_results", {}).setdefault("nonlinear_metrics", {})["recurrence_rate"] = rqa_meta.get("Recurrence Rate (RR)", 0.0)
            fig_rp = plot_recurrence_matrix(r_mat, title="Binary Recurrence Matrix R(i, j)")
            st.plotly_chart(fig_rp, use_container_width=True)

        with c_r2:
            st.markdown("**Recurrence Quantification Analysis (RQA) Metrics**")
            rqa_table = pd.DataFrame([
                {"RQA Parameter": k, "Observed Value": f"{v:.4f}" if isinstance(v, float) else str(v)}
                for k, v in rqa_meta.items()
            ])
            st.dataframe(rqa_table, use_container_width=True, hide_index=True)
            st.caption("Diagonal lines indicate deterministic predictability; vertical structures reflect laminar trapping states; isolated points reflect stochastic scattering.")

