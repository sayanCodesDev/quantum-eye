"""
QUANTUM EYE — Model Explainability Engine (SHAP)
Computes SHAP feature importance, summary values, and local prediction explanations.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, Tuple, Optional, List

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def compute_shap_explanations(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    max_evals: int = 100
) -> Dict[str, Any]:
    """
    Computes SHAP values using TreeExplainer (if tree-based) or Kernel/SamplingExplainer.
    Returns feature importance rankings and values.
    """
    if not SHAP_AVAILABLE:
        return {"error": "SHAP library is not installed in the environment."}

    # Limit sample size for responsive computation
    n_samples = min(len(X_test), 50)
    X_sub = X_test[:n_samples]
    X_bg = X_train[:min(len(X_train), 50)]

    try:
        # Check if model has tree structures
        model_type = type(model).__name__.lower()
        if "forest" in model_type or "tree" in model_type or "gradient" in model_type:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sub)
        else:
            explainer = shap.KernelExplainer(model.predict_proba if hasattr(model, "predict_proba") else model.predict, X_bg)
            shap_values = explainer.shap_values(X_sub, nsamples=max_evals)

        # Handle multiclass or list of arrays
        if isinstance(shap_values, list):
            # Take mean absolute SHAP across all classes
            mean_abs_per_class = [np.mean(np.abs(sv), axis=0) for sv in shap_values]
            global_importance = np.mean(mean_abs_per_class, axis=0)
            sample_shap = shap_values[0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            global_importance = np.mean(np.abs(shap_values), axis=(0, 2))
            sample_shap = shap_values[:, :, 0]
        else:
            global_importance = np.mean(np.abs(shap_values), axis=0)
            sample_shap = shap_values

        # Rank features
        ranking_indices = np.argsort(global_importance)[::-1]
        ranked_features = [feature_names[i] for i in ranking_indices]
        ranked_scores = [float(global_importance[i]) for i in ranking_indices]

        return {
            "ranked_features": ranked_features,
            "ranked_scores": ranked_scores,
            "global_importance": global_importance,
            "shap_values": sample_shap,
            "X_sub": X_sub,
            "feature_names": feature_names
        }
    except Exception as e:
        return {"error": f"SHAP calculation encountered error: {str(e)}"}


def plot_shap_importance(ranked_features: List[str], ranked_scores: List[float], top_k: int = 12) -> go.Figure:
    """Plots interactive horizontal bar chart of top SHAP feature importances."""
    fig = go.Figure()

    top_feats = ranked_features[:top_k][::-1]
    top_vals = ranked_scores[:top_k][::-1]

    fig.add_trace(go.Bar(
        x=top_vals,
        y=top_feats,
        orientation='h',
        marker=dict(
            color=top_vals,
            colorscale='Blues',
            line=dict(color='#0284c7', width=1)
        ),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text=f"Global Feature Importance (Top {len(top_feats)} SHAP Attributes)", font=dict(size=15, color="#0f172a")),
        xaxis=dict(title="Mean Absolute SHAP Value (Impact on Model Output)", gridcolor="#f1f5f9"),
        yaxis=dict(title="Feature Name", gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=150, r=20, t=50, b=40),
        height=400
    )
    return fig

