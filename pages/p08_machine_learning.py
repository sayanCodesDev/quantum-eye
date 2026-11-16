"""
QUANTUM EYE — Page 8: Machine Learning Engine
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from modules.state_manager import get_current_signal
from modules.ml_engine import MLEngine
from modules.feature_extraction import extract_windowed_feature_matrix

def render():
    st.title("🤖 Machine Learning Intelligence Engine")
    st.markdown("Execute Supervised Classification, Unsupervised Clustering & PCA Dimensionality Reduction, and the controlled 3-Class Signal Intelligence benchmark.")

    prep = st.session_state.get("preprocessor")
    if prep is None:
        st.warning("No dataset loaded. Please visit 'Upload Dataset' first.")
        return

    df_proc = prep.processed_df
    num_cols = list(df_proc.select_dtypes(include=[np.number]).columns)
    all_cols = list(df_proc.columns)

    tab_exp, tab_sup, tab_unsup = st.tabs([
        "🔬 3-Class Signal Intelligence Experiment",
        "🎯 Supervised Classification (Dataset)",
        "🪐 Unsupervised Clustering & PCA"
    ])

    # ----------------- 1. 3-CLASS CONTROLLED EXPERIMENT -----------------
    with tab_exp:
        st.subheader("Controlled 3-Class Signal Classification Experiment")
        st.markdown("""
        Evaluates classifiers across independently generated signals from three fundamental physical regimes:
        1. **Periodic signals** (Harmonics, Sine, Square, Sawtooth)
        2. **Stochastic signals** (Gaussian White Noise, Pink Noise, Brown Noise)
        3. **Deterministic Chaotic signals** (Lorenz Attractor, Logistic Map, Rössler)
        *Guarantees zero temporal data-leakage across train and test sets.*
        """)

        c_exp1, c_exp2, c_exp3 = st.columns(3)
        with c_exp1:
            n_per_class = st.slider("Signals per Class", min_value=10, max_value=60, value=25, step=5)
        with c_exp2:
            sig_len = st.slider("Samples per Signal", min_value=200, max_value=1000, value=500, step=100)
        with c_exp3:
            noise_snr = st.selectbox("Noise Level (SNR)", [None, 30.0, 20.0, 15.0, 10.0, 5.0],
                                     format_func=lambda x: "No Noise (Clean)" if x is None else f"{x} dB SNR")

        c_m1, c_m2 = st.columns(2)
        with c_m1:
            exp_model = st.selectbox("Classifier Algorithm", ["Random Forest", "Support Vector Machine (RBF)", "Gradient Boosting", "Decision Tree", "K-Nearest Neighbors"])
        with c_m2:
            test_split = st.slider("Test Split Ratio", min_value=0.15, max_value=0.40, value=0.25, step=0.05)

        if st.button("🚀 Run 3-Class Intelligence Benchmark", type="primary"):
            with st.spinner("Generating signals, extracting feature matrices, and evaluating classifier..."):
                X_exp, y_exp = MLEngine.run_signal_intelligence_experiment(
                    n_signals_per_class=n_per_class,
                    signal_length=sig_len,
                    fs=100.0,
                    noise_snr_db=noise_snr
                )
                res = MLEngine.train_and_evaluate_supervised(
                    X=X_exp.values,
                    y=y_exp.values,
                    model_name=exp_model,
                    test_size=test_split,
                    feature_names=list(X_exp.columns)
                )
                st.session_state["ml_results"] = res
                st.session_state["ml_dataset"] = (X_exp, y_exp)

        if "ml_results" in st.session_state and st.session_state["ml_results"] is not None:
            res = st.session_state["ml_results"]
            st.markdown("---")
            st.subheader(f"📊 Benchmark Performance: `{res['model_name']}`")
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Classification Accuracy", f"{res['accuracy']*100:.2f}%")
            k2.metric("Weighted F1-Score", f"{res['f1_score']:.4f}")
            k3.metric("Precision", f"{res['precision']:.4f}")
            k4.metric("Recall", f"{res['recall']:.4f}")

            # Confusion Matrix
            col_cm, col_time = st.columns([1.2, 1])
            with col_cm:
                st.markdown("**Confusion Matrix**")
                cm = res["confusion_matrix"]
                labels = res["classes"]
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=[f"Pred: {c}" for c in labels],
                    y=[f"True: {c}" for c in labels],
                    colorscale='Blues',
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 14}
                ))
                fig_cm.update_layout(
                    margin=dict(l=40, r=20, t=30, b=40),
                    height=320,
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_time:
                st.markdown("**Computational Telemetry**")
                st.markdown(f"""
                <div class="qe-card qe-card-teal">
                    <p><b>Training Execution Time:</b> <code>{res['train_time_sec']:.4f} s</code></p>
                    <p><b>Inference Latency:</b> <code>{res['inference_time_sec']*1000:.2f} ms</code></p>
                    <p><b>Cross-Validation Mean:</b> <code>{res['cv_mean']*100:.2f}% (±{res['cv_std']*100:.2f}%)</code></p>
                    <p><b>Feature Dimensionality:</b> <code>{len(res['feature_names'])} attributes</code></p>
                </div>
                """, unsafe_allow_html=True)

    # ----------------- 2. SUPERVISED ON ACTIVE DATASET -----------------
    with tab_sup:
        st.subheader("Train Supervised Classifier on Dataset")
        st.markdown("Train a model on your uploaded or benchmark dataset using a designated target label column.")

        c_target, c_features = st.columns([1, 2])
        with c_target:
            target_col = st.selectbox("Target Label Column", all_cols, index=len(all_cols)-1)
        with c_features:
            eligible_feats = [c for c in num_cols if c != target_col]
            selected_feats = st.multiselect("Predictor Features", eligible_feats, default=eligible_feats)

        if len(selected_feats) == 0:
            st.warning("Please select at least one numerical predictor feature.")
        else:
            c_s1, c_s2, c_s3 = st.columns(3)
            with c_s1:
                sup_model = st.selectbox("Model", ["Random Forest", "Support Vector Machine", "Gradient Boosting", "K-Nearest Neighbors", "Decision Tree"])
            with c_s2:
                sup_test_size = st.slider("Test Size", 0.1, 0.5, 0.25, 0.05)
            with c_s3:
                sup_cv = st.slider("CV Folds", 2, 10, 5)

            if st.button("Train Custom Classifier", type="primary"):
                with st.spinner("Training model with standard scaling (no leakage)..."):
                    sub_df = df_proc.dropna(subset=[target_col] + selected_feats)
                    X_custom = sub_df[selected_feats].values
                    y_custom = sub_df[target_col].values

                    res_custom = MLEngine.train_and_evaluate_supervised(
                        X=X_custom,
                        y=y_custom,
                        model_name=sup_model,
                        test_size=sup_test_size,
                        cv_folds=sup_cv,
                        feature_names=selected_feats
                    )
                    st.session_state["ml_results"] = res_custom
                    st.success(f"Trained {sup_model}! Test Accuracy: {res_custom['accuracy']*100:.2f}%, F1: {res_custom['f1_score']:.4f}")

    # ----------------- 3. UNSUPERVISED LEARNING & PCA -----------------
    with tab_unsup:
        st.subheader("Unsupervised Pattern Discovery & PCA Projection")
        if len(num_cols) < 2:
            st.info("Unsupervised exploration requires at least 2 numerical features.")
        else:
            unsup_feats = st.multiselect("Features for Clustering & PCA", num_cols, default=num_cols[:min(8, len(num_cols))])
            if len(unsup_feats) >= 2:
                X_unsup = df_proc[unsup_feats].dropna().values

                col_pca, col_clus = st.columns(2)
                with col_pca:
                    st.markdown("**Principal Component Analysis (PCA)**")
                    pca_res = MLEngine.run_pca(X_unsup, n_components=3)
                    st.write(f"Total Variance Explained (Top 3): **{pca_res['total_variance_explained']*100:.2f}%**")
                    
                    X_pca = pca_res["X_pca"]
                    fig_pca = go.Figure()
                    fig_pca.add_trace(go.Scatter(
                        x=X_pca[:, 0],
                        y=X_pca[:, 1],
                        mode='markers',
                        marker=dict(color='#0284c7', size=5, opacity=0.7),
                        name='PCA Samples'
                    ))
                    fig_pca.update_layout(
                        title=f"2D PCA Projection ({pca_res['explained_variance_ratio'][0]*100:.1f}% & {pca_res['explained_variance_ratio'][1]*100:.1f}%)",
                        xaxis=dict(title="Principal Component 1", gridcolor="#f1f5f9"),
                        yaxis=dict(title="Principal Component 2", gridcolor="#f1f5f9"),
                        plot_bgcolor="#ffffff",
                        paper_bgcolor="#ffffff",
                        height=350
                    )
                    st.plotly_chart(fig_pca, use_container_width=True)

                with col_clus:
                    st.markdown("**K-Means Clustering**")
                    k_clusters = st.slider("Cluster Count (k)", min_value=2, max_value=8, value=3)
                    km_res = MLEngine.run_kmeans(X_unsup, n_clusters=k_clusters)
                    
                    fig_km = go.Figure()
                    fig_km.add_trace(go.Scatter(
                        x=X_pca[:, 0],
                        y=X_pca[:, 1],
                        mode='markers',
                        marker=dict(color=km_res["cluster_labels"], colorscale='Viridis', size=6, opacity=0.8),
                        name='Clusters'
                    ))
                    fig_km.update_layout(
                        title=f"K-Means (k={k_clusters}) on PCA Space",
                        xaxis=dict(title="PC 1", gridcolor="#f1f5f9"),
                        yaxis=dict(title="PC 2", gridcolor="#f1f5f9"),
                        plot_bgcolor="#ffffff",
                        paper_bgcolor="#ffffff",
                        height=350
                    )
                    st.plotly_chart(fig_km, use_container_width=True)
                    st.caption(f"Clustering Inertia: {km_res['inertia']:.2f}. Clusters represent mathematical groupings without assigned physical causality.")

