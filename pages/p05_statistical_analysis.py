"""
QUANTUM EYE — Page 5: Statistical & Correlation Analysis
"""
import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal, stats
import plotly.graph_objects as go
from modules.state_manager import get_current_signal
from modules.nonlinear_analysis import compute_shannon_entropy

def render():
    st.title("📊 Statistical Dynamics & Correlation Analysis")
    st.markdown("Quantify serial temporal dependence (autocorrelation), information entropy, cross-correlations, and multivariate signal matrices.")

    time_axis, signal_values, sig_name, fs = get_current_signal()
    prep = st.session_state.get("preprocessor")
    if signal_values is None or prep is None:
        st.warning("No signal loaded. Please select or upload a dataset.")
        return

    df_proc = prep.processed_df
    num_cols = list(df_proc.select_dtypes(include=[np.number]).columns)

    tab_acf, tab_cross, tab_heat, tab_roll_ent = st.tabs([
        "🔄 Autocorrelation (ACF)", "🔀 Cross-Correlation", "🔥 Multivariate Correlation Heatmap", "📈 Rolling Shannon Entropy"
    ])

    # 1. Autocorrelation Function (ACF)
    with tab_acf:
        st.subheader(f"Serial Autocorrelation Function: `{sig_name}`")
        max_lag = st.slider("Maximum Lag Steps", min_value=10, max_value=min(1000, len(signal_values)-1), value=min(150, len(signal_values)//3))
        
        # Calculate normalized autocorrelation
        norm_s = signal_values - np.mean(signal_values)
        s_std = np.std(signal_values)
        if s_std > 1e-9:
            norm_s = norm_s / s_std
            # Full correlation
            autocorr_full = signal.correlate(norm_s, norm_s, mode='full') / len(norm_s)
            mid = len(norm_s) - 1
            lags = np.arange(0, max_lag + 1)
            acf_vals = autocorr_full[mid : mid + max_lag + 1]
        else:
            lags = np.arange(0, max_lag + 1)
            acf_vals = np.zeros(len(lags))

        # Bartlett's 95% confidence intervals (+/- 1.96 / sqrt(N))
        conf_95 = 1.96 / np.sqrt(len(signal_values))

        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(
            x=lags,
            y=acf_vals,
            name='Autocorrelation',
            marker=dict(color='#0284c7')
        ))
        # Upper and lower confidence lines
        fig_acf.add_hline(y=conf_95, line_dash="dash", line_color="#ef4444", annotation_text="95% CI (+)", annotation_position="top right")
        fig_acf.add_hline(y=-conf_95, line_dash="dash", line_color="#ef4444", annotation_text="95% CI (-)", annotation_position="bottom right")

        fig_acf.update_layout(
            title=dict(text=f"Autocorrelation vs. Lag (k) — {sig_name}", font=dict(size=15, color="#0f172a")),
            xaxis=dict(title=f"Lag Steps ({f'x {1/fs:.3f}s' if fs else 'samples'})", gridcolor="#f1f5f9"),
            yaxis=dict(title="Correlation Coefficient r(k)", gridcolor="#f1f5f9", range=[-1.05, 1.05]),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            margin=dict(l=40, r=20, t=50, b=40),
            height=380
        )
        st.plotly_chart(fig_acf, use_container_width=True)

        # Autocorrelation decay summary
        zero_crossings = np.where(np.diff(np.sign(acf_vals)))[0]
        first_zero = lags[zero_crossings[0]] if len(zero_crossings) > 0 else max_lag
        st.caption(f"First autocorrelation zero-crossing at lag **{first_zero}** samples. Slow decay indicates long-range memory or non-stationarity; rapid drop indicates stochastic white noise.")

    # 2. Cross-Correlation between 2 signals
    with tab_cross:
        st.subheader("Bivariate Cross-Correlation")
        if len(num_cols) < 2:
            st.info("Cross-correlation requires at least two numerical columns in the dataset.")
        else:
            col_x1, col_x2 = st.columns(2)
            with col_x1:
                sig_1 = st.selectbox("Signal 1 (Reference)", num_cols, index=num_cols.index(sig_name) if sig_name in num_cols else 0)
            with col_x2:
                eligible_2 = [c for c in num_cols if c != sig_1]
                sig_2 = st.selectbox("Signal 2 (Target)", eligible_2 if eligible_2 else num_cols, index=0)

            s1 = df_proc[sig_1].dropna().values
            s2 = df_proc[sig_2].dropna().values
            min_l = min(len(s1), len(s2))
            s1 = s1[:min_l]
            s2 = s2[:min_l]

            # Cross correlation
            s1_norm = (s1 - np.mean(s1)) / (np.std(s1) + 1e-9)
            s2_norm = (s2 - np.mean(s2)) / (np.std(s2) + 1e-9)

            max_x_lag = st.slider("Cross-Correlation Max Lag", min_value=10, max_value=min(500, min_l - 1), value=min(100, min_l // 3))
            xcorr = signal.correlate(s1_norm, s2_norm, mode='full') / min_l
            mid = min_l - 1
            x_lags = np.arange(-max_x_lag, max_x_lag + 1)
            xcorr_sub = xcorr[mid - max_x_lag : mid + max_x_lag + 1]

            peak_lag = x_lags[np.argmax(np.abs(xcorr_sub))]
            peak_val = xcorr_sub[np.argmax(np.abs(xcorr_sub))]

            fig_cc = go.Figure()
            fig_cc.add_trace(go.Scatter(
                x=x_lags,
                y=xcorr_sub,
                mode='lines',
                line=dict(color='#0d9488', width=2),
                name='Cross-Correlation'
            ))
            fig_cc.add_vline(x=peak_lag, line_dash="dash", line_color="#ef4444", annotation_text=f"Peak Lag: {peak_lag}", annotation_position="top right")

            fig_cc.update_layout(
                title=dict(text=f"Cross-Correlation: '{sig_1}' vs '{sig_2}' (Peak: {peak_val:.3f} at lag {peak_lag})", font=dict(size=15, color="#0f172a")),
                xaxis=dict(title="Lag Offset", gridcolor="#f1f5f9"),
                yaxis=dict(title="Cross-Correlation", gridcolor="#f1f5f9"),
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                margin=dict(l=40, r=20, t=50, b=40),
                height=380
            )
            st.plotly_chart(fig_cc, use_container_width=True)

    # 3. Multivariate Correlation Heatmap
    with tab_heat:
        st.subheader("Multivariate Feature Correlation Matrix")
        if len(num_cols) < 2:
            st.info("Multivariate correlation requires at least 2 numerical columns.")
        else:
            corr_method = st.radio("Correlation Method", ["pearson", "spearman"], format_func=lambda x: "Pearson (Linear)" if x == "pearson" else "Spearman (Rank-Monotonic)", horizontal=True)
            corr_matrix = df_proc[num_cols].corr(method=corr_method).round(3)

            fig_h = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu_r',
                zmin=-1.0,
                zmax=1.0,
                text=corr_matrix.values,
                texttemplate="%{text}",
                textfont={"size": 10}
            ))

            fig_h.update_layout(
                title=dict(text=f"Pairwise Correlation Matrix ({corr_method.capitalize()})", font=dict(size=15, color="#0f172a")),
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                margin=dict(l=60, r=20, t=50, b=60),
                height=450
            )
            st.plotly_chart(fig_h, use_container_width=True)

    # 4. Rolling Shannon Entropy
    with tab_roll_ent:
        st.subheader("Rolling Shannon Information Entropy Trajectory")
        r_win = st.slider("Rolling Entropy Window", min_value=20, max_value=min(1000, len(signal_values)-1), value=min(100, len(signal_values)//4))
        
        # Calculate rolling Shannon entropy
        n_pts = len(signal_values)
        stride = max(1, n_pts // 1000)
        roll_times = []
        roll_entropies = []

        for i in range(0, n_pts - r_win + 1, stride):
            sub_w = signal_values[i : i + r_win]
            h = compute_shannon_entropy(sub_w, num_bins=20)
            roll_times.append(time_axis[i + r_win // 2])
            roll_entropies.append(h)

        fig_re = go.Figure()
        fig_re.add_trace(go.Scatter(
            x=roll_times,
            y=roll_entropies,
            mode='lines',
            line=dict(color='#8b5cf6', width=2),
            name='Rolling Shannon Entropy',
            fill='tozeroy',
            fillcolor='rgba(139, 92, 246, 0.1)'
        ))

        fig_re.update_layout(
            title=dict(text=f"Rolling Shannon Entropy (Window = {r_win} samples)", font=dict(size=15, color="#0f172a")),
            xaxis=dict(title=f"Time ({'s' if fs else 'Sample Index'})", gridcolor="#f1f5f9"),
            yaxis=dict(title="Entropy (Bits)", gridcolor="#f1f5f9"),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            margin=dict(l=40, r=20, t=50, b=40),
            height=380
        )
        st.plotly_chart(fig_re, use_container_width=True)

