"""
QUANTUM EYE — Scientific LLM & Rule-Based Interpretation Layer
Interprets structured mathematical and machine learning findings.
Provides an automated scientific rule-based explanation engine when no API key is provided,
and supports optional Gemini/OpenAI API integration without hallucinating numbers.
"""
import os
import json
import requests
from typing import Dict, Any, Optional

class ScientificInterpreter:
    """Interprets complex signal analysis findings into rigorous scientific prose."""

    @staticmethod
    def generate_explanation(
        analysis_context: Dict[str, Any],
        api_key: Optional[str] = None,
        provider: str = "gemini"
    ) -> str:
        """
        Generates explanation using LLM API if key is provided,
        or falls back to the deterministic scientific rule-based reasoning engine.
        """
        if api_key and len(api_key.strip()) > 5:
            try:
                if provider.lower() == "gemini":
                    return ScientificInterpreter._call_gemini_api(analysis_context, api_key.strip())
                elif provider.lower() == "openai":
                    return ScientificInterpreter._call_openai_api(analysis_context, api_key.strip())
            except Exception as e:
                # On API error, smoothly fallback with warning
                fallback = ScientificInterpreter._rule_based_interpretation(analysis_context)
                return f"> [!WARNING]\n> LLM API call failed ({str(e)}). Displaying fallback deterministic scientific assessment below.\n\n" + fallback

        return ScientificInterpreter._rule_based_interpretation(analysis_context)

    @staticmethod
    def _rule_based_interpretation(ctx: Dict[str, Any]) -> str:
        """
        Deterministic, mathematically grounded scientific interpreter.
        Guarantees zero hallucinated measurements.
        """
        sig_name = ctx.get("signal_name", "Analyzed Signal")
        n_samples = ctx.get("sample_count", 0)
        mean_val = ctx.get("mean", 0.0)
        std_val = ctx.get("std", 0.0)
        skew_val = ctx.get("skewness", 0.0)
        kurt_val = ctx.get("kurtosis", 0.0)
        dom_freq = ctx.get("dominant_freq", 0.0)
        freq_unit = ctx.get("freq_unit", "Hz")
        spec_entropy = ctx.get("spectral_entropy", 0.0)
        norm_spec_entropy = ctx.get("norm_spectral_entropy", 0.0)
        shannon_ent = ctx.get("shannon_entropy", 0.0)
        perm_ent = ctx.get("perm_entropy", 0.0)
        sample_ent = ctx.get("sample_entropy", 0.0)
        lyapunov = ctx.get("lyapunov_exponent", None)
        n_anomalies = ctx.get("anomaly_count", 0)
        ml_prediction = ctx.get("ml_prediction", None)
        ml_model = ctx.get("ml_model", "Machine Learning Classifier")
        ml_accuracy = ctx.get("ml_accuracy", None)
        quantum_accuracy = ctx.get("quantum_accuracy", None)
        top_features = ctx.get("top_features", [])

        # 1. Signal Characterization
        lines = []
        lines.append(f"### 🔬 Scientific Synthesis for '{sig_name}'\n")
        lines.append(f"**Data Profile**: {n_samples:,} samples | $\\mu = {mean_val:.4f}$ | $\\sigma = {std_val:.4f}$ | Skewness: `{skew_val:.3f}` | Excess Kurtosis: `{kurt_val:.3f}`\n")

        # 2. Spectral Assessment
        lines.append("#### 1. Frequency Dynamics & Periodicity")
        if dom_freq > 0:
            lines.append(f"- Dominant spectral peak observed at **{dom_freq:.3f} {freq_unit}**.")
        if norm_spec_entropy < 0.35:
            lines.append("- **Spectral Concentration**: Low normalized spectral entropy indicates sharp harmonic concentration, characteristic of periodic or quasi-periodic oscillations.")
        elif norm_spec_entropy > 0.75:
            lines.append("- **Broadband Dispersion**: High spectral entropy reflects flat power distribution across frequencies, strongly indicative of stochastic noise or broad chaotic resonance.")
        else:
            lines.append("- **Mixed Bandwidth**: Intermediate spectral distribution with identifiable resonances amidst continuous background energy.")

        # 3. Nonlinear Dynamics & Information Theory
        lines.append("\n#### 2. Nonlinear Dynamics & Entropy Analysis")
        lines.append(f"- **Shannon Amplitude Entropy**: `{shannon_ent:.3f}` bits.")
        lines.append(f"- **Permutation Entropy (Normalized)**: `{perm_ent:.3f}` (quantifies temporal ordinal pattern disorder).")
        if sample_ent > 0:
            lines.append(f"- **Sample Entropy ($m=2$)**: `{sample_ent:.3f}` (measures conditional probability of repeating trajectories).")

        if lyapunov is not None:
            if lyapunov > 0.05:
                lines.append(f"- **Largest Lyapunov Exponent ($\\lambda_1$)**: `{lyapunov:.4f}`. Positive divergence rate confirms exponential trajectory separation in reconstructed phase space, consistent with deterministic chaos or stochastic divergence.")
            elif abs(lyapunov) <= 0.05:
                lines.append(f"- **Largest Lyapunov Exponent ($\\lambda_1$)**: `{lyapunov:.4f}`. Near-zero divergence indicates stable limit cycles or conservative quasi-periodic motion.")
            else:
                lines.append(f"- **Largest Lyapunov Exponent ($\\lambda_1$)**: `{lyapunov:.4f}`. Negative exponent reflects trajectory convergence toward a stable fixed-point attractor.")

        # 4. Anomaly Profile
        lines.append("\n#### 3. Anomaly & Outlier Distribution")
        if n_anomalies == 0:
            lines.append("- No statistical deviations exceeded the selected operational threshold. The series exhibits stationarity within established confidence bounds.")
        else:
            anomaly_pct = (n_anomalies / max(1, n_samples)) * 100
            lines.append(f"- Detected **{n_anomalies} anomalies** ({anomaly_pct:.2f}% of samples).")
            lines.append("- *Scientific Note*: In complex signals, statistical anomalies indicate transient deviations or non-stationary shocks; they do not automatically denote operational faults without domain cross-validation.")

        # 5. Machine Learning & Quantum Intelligence
        if ml_prediction or ml_accuracy is not None:
            lines.append("\n#### 4. Machine Learning & Quantum Inference")
            if ml_prediction:
                lines.append(f"- **Predicted Signal Classification**: **{ml_prediction}** (Model: {ml_model})")
            if ml_accuracy is not None:
                lines.append(f"- Classical Model Accuracy: **{ml_accuracy * 100:.2f}%**")
            if quantum_accuracy is not None:
                lines.append(f"- Quantum Kernel SVM Accuracy: **{quantum_accuracy * 100:.2f}%**")
            if top_features:
                feat_str = ", ".join([f"`{f}`" for f in top_features[:5]])
                lines.append(f"- **Most Influential Predictive Features (SHAP)**: {feat_str}")

        # 6. Mathematical Limitations
        lines.append("\n#### 5. Methodological Limitations & Transparency")
        lines.append("- Finite sample lengths and observational noise can bias entropy estimates and false-positive Lyapunov divergence.")
        lines.append("- Classical vs. Quantum comparisons are executed on exact state-vector simulators (`default.qubit`); physical QPUs are subject to decoherence and gate noise.")
        lines.append("- No single metric proves deterministic chaos in isolation; joint validation across phase space, spectrum, and permutation entropy is required.")

        return "\n".join(lines)

    @staticmethod
    def _call_gemini_api(ctx: Dict[str, Any], api_key: str) -> str:
        """Invokes Google Gemini API with strict temperature to prevent hallucination."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        prompt = (
            "You are QUANTUM EYE's scientific intelligence agent. Analyze the following verified signal analysis "
            "measurements. Explain the findings with rigorous scientific precision. DO NOT invent or alter any numbers. "
            "Acknowledge the physical and mathematical limitations of the methods.\n\n"
            f"Measurements JSON:\n{json.dumps(ctx, indent=2)}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 800}
        }
        res = requests.post(url, json=payload, timeout=15)
        res.raise_for_status()
        data = res.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    @staticmethod
    def _call_openai_api(ctx: Dict[str, Any], api_key: str) -> str:
        """Invokes OpenAI API."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        prompt = (
            "You are QUANTUM EYE's scientific intelligence agent. Analyze the following verified signal analysis "
            "measurements. Explain the findings with rigorous scientific precision. DO NOT invent or alter any numbers. "
            "Acknowledge the physical and mathematical limitations of the methods.\n\n"
            f"Measurements JSON:\n{json.dumps(ctx, indent=2)}"
        )
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
        res = requests.post(url, headers=headers, json=payload, timeout=15)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"]

