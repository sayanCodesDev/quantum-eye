"""
QUANTUM EYE — Page 12: Model Explainability (SHAP), LLM Interpretation & PDF Report
"""
import streamlit as st
import numpy as np
import pandas as pd
from modules.state_manager import get_current_signal
from modules.explainability import compute_shap_explanations, plot_shap_importance
from modules.llm_interpreter import ScientificInterpreter
from modules.report_generator import generate_pdf_report

def render():
    st.title("📑 Explainability, Scientific Interpretation & Reports")
    st.markdown("Inspect SHAP feature importances, generate mathematically grounded scientific explanations, and compile a comprehensive laboratory PDF report.")

    time_axis, signal_values, sig_name, fs = get_current_signal()
    prep = st.session_state.get("preprocessor")
    ml_res = st.session_state.get("ml_results", None)
    q_res = st.session_state.get("quantum_results", None)
    cached = st.session_state.get("cached_results", {})

    tab_shap, tab_llm, tab_pdf = st.tabs([
        "🔍 SHAP Model Explainability",
        "🧠 Scientific Synthesis & AI Interpretation",
        "📄 Export Laboratory PDF Report"
    ])

    # ----------------- 1. SHAP EXPLAINABILITY -----------------
    with tab_shap:
        st.subheader("SHAP (SHapley Additive exPlanations) Feature Attribution")
        if ml_res is None or "model" not in ml_res:
            st.info("Train a model in the **Machine Learning** section to calculate SHAP feature contributions.")
        else:
            with st.spinner("Computing SHAP Shapley values..."):
                shap_data = compute_shap_explanations(
                    model=ml_res["model"],
                    X_train=ml_res["X_train"],
                    X_test=ml_res["X_test"],
                    feature_names=ml_res["feature_names"]
                )

            if "error" in shap_data:
                st.warning(shap_data["error"])
            else:
                top_k = st.slider("Display Top Features", min_value=5, max_value=min(25, len(shap_data["ranked_features"])), value=10)
                fig_shap = plot_shap_importance(shap_data["ranked_features"], shap_data["ranked_scores"], top_k=top_k)
                st.plotly_chart(fig_shap, use_container_width=True)

                st.markdown("**Top Feature Attribution Table**")
                shap_df = pd.DataFrame({
                    "Rank": range(1, len(shap_data["ranked_features"]) + 1),
                    "Feature Attribute": shap_data["ranked_features"],
                    "Mean |SHAP| Impact": np.round(shap_data["ranked_scores"], 5)
                }).head(top_k)
                st.dataframe(shap_df, use_container_width=True, hide_index=True)
                st.session_state["top_features"] = shap_data["ranked_features"][:5]

    # ----------------- 2. LLM & RULE-BASED INTERPRETATION -----------------
    with tab_llm:
        st.subheader("Scientific Synthesis & Mathematical Interpretation")
        st.markdown("""
        Quantum Eye provides a deterministic scientific rule-based interpretation engine grounded in verified calculations.
        Optionally, you may supply a Google Gemini or OpenAI API key to generate conversational scientific prose without hallucinating numbers.
        """)

        c_prov, c_key = st.columns([1, 2])
        with c_prov:
            prov = st.selectbox("Interpretation Engine", ["Scientific Rule-Based Engine (Built-in Fallback)", "Google Gemini API", "OpenAI API"])
        with c_key:
            if "Gemini" in prov:
                key_input = st.text_input("Gemini API Key (Optional)", type="password", placeholder="AIzaSy...")
            elif "OpenAI" in prov:
                key_input = st.text_input("OpenAI API Key (Optional)", type="password", placeholder="sk-...")
            else:
                key_input = ""
                st.caption("Using built-in deterministic scientific reasoning engine (no network required).")

        # Assemble structured analysis context
        td = cached.get("time_domain_metrics", {})
        fd = cached.get("spectral_metrics", {})
        nl = cached.get("nonlinear_metrics", {})
        ad = cached.get("anomaly_results", {})

        analysis_context = {
            "signal_name": sig_name or "Signal",
            "sample_count": len(signal_values) if signal_values is not None else 0,
            "mean": td.get("Mean", 0.0),
            "std": td.get("Std Dev", 0.0),
            "skewness": td.get("Skewness", 0.0),
            "kurtosis": td.get("Excess Kurtosis", 0.0),
            "dominant_freq": fd.get("dominant_frequency", 0.0),
            "freq_unit": fd.get("freq_unit", "Hz"),
            "spectral_entropy": fd.get("spectral_entropy", 0.0),
            "norm_spectral_entropy": fd.get("normalized_spectral_entropy", 0.0),
            "shannon_entropy": nl.get("shannon_entropy", 0.0),
            "perm_entropy": nl.get("perm_entropy", 0.0),
            "sample_entropy": nl.get("sample_entropy", 0.0),
            "lyapunov_exponent": nl.get("lyapunov_exponent", None),
            "anomaly_count": ad.get("anomaly_count", 0),
            "ml_prediction": ml_res.get("y_pred", [None])[0] if ml_res else None,
            "ml_model": ml_res.get("model_name", "None") if ml_res else "None",
            "ml_accuracy": ml_res.get("accuracy", None) if ml_res else None,
            "quantum_accuracy": q_res.get("quantum", {}).get("accuracy", None) if q_res else None,
            "top_features": st.session_state.get("top_features", [])
        }

        if st.button("Generate Scientific Synthesis", type="primary"):
            with st.spinner("Synthesizing multi-domain findings..."):
                target_prov = "gemini" if "Gemini" in prov else ("openai" if "OpenAI" in prov else "fallback")
                synthesis_text = ScientificInterpreter.generate_explanation(
                    analysis_context=analysis_context,
                    api_key=key_input,
                    provider=target_prov
                )
                st.session_state["interpretation_text"] = synthesis_text

        if "interpretation_text" in st.session_state:
            st.markdown(st.session_state["interpretation_text"])

    # ----------------- 3. PDF REPORT GENERATION -----------------
    with tab_pdf:
        st.subheader("Generate Downloadable Scientific Laboratory PDF")
        st.markdown("Compiles all metrics, preprocessing steps, spectral properties, nonlinear dynamics, anomaly counts, and model comparisons into a publication-ready PDF document.")

        if st.button("📑 Compile Scientific PDF Report", type="primary"):
            with st.spinner("Compiling ReportLab document..."):
                report_payload = {
                    "dataset_name": st.session_state.get("dataset_name", "Quantum Eye Dataset"),
                    "sample_count": len(signal_values) if signal_values is not None else 0,
                    "signal_column": sig_name or "Signal",
                    "sampling_rate": fs if fs else "Unknown",
                    "preprocessing_history": prep.pipeline_history if prep else [],
                    "time_domain_metrics": cached.get("time_domain_metrics", {}),
                    "spectral_metrics": cached.get("spectral_metrics", {}),
                    "nonlinear_metrics": cached.get("nonlinear_metrics", {}),
                    "anomaly_results": cached.get("anomaly_results", {}),
                    "ml_results": ml_res or {},
                    "quantum_results": q_res or {},
                    "interpretation_text": st.session_state.get("interpretation_text", "")
                }
                pdf_bytes = generate_pdf_report(report_payload)
                st.session_state["pdf_bytes"] = pdf_bytes
                st.success("PDF Report generated successfully!")

        if "pdf_bytes" in st.session_state:
            st.download_button(
                label="⬇️ Download Laboratory Report (PDF)",
                data=st.session_state["pdf_bytes"],
                file_name=f"quantum_eye_report_{sig_name or 'signal'}.pdf",
                mime="application/pdf"
            )

