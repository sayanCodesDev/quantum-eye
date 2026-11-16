"""
QUANTUM EYE — Anomaly Detection Engine
Implements Isolation Forest, Z-Score, Rolling Deviation, Median Absolute Deviation (MAD),
and One-Class SVM with interactive Plotly visualizers and sensitivity controls.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import plotly.graph_objects as go
from typing import Dict, Any, Tuple, Optional

class AnomalyDetector:
    """Scientific anomaly detection engine."""

    @staticmethod
    def detect_zscore(signal_values: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """Z-Score thresholding: score = |(x - mu) / sigma|."""
        clean = np.nan_to_num(signal_values)
        mu = np.mean(clean)
        sigma = np.std(clean)
        if sigma < 1e-9:
            scores = np.zeros(len(clean))
        else:
            scores = np.abs((clean - mu) / sigma)
        is_anomaly = scores > threshold
        return is_anomaly, scores

    @staticmethod
    def detect_rolling_deviation(signal_values: np.ndarray, window: int = 50, k: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """Rolling window statistical thresholding: score = |x - rolling_mean| / rolling_std."""
        s = pd.Series(signal_values)
        r_mean = s.rolling(window=window, min_periods=1, center=True).mean()
        r_std = s.rolling(window=window, min_periods=1, center=True).std().replace(0, 1e-9).bfill().ffill()
        scores = np.abs((s - r_mean) / r_std).fillna(0).values
        is_anomaly = scores > k
        return is_anomaly, scores

    @staticmethod
    def detect_mad(signal_values: np.ndarray, threshold: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
        """Median Absolute Deviation (MAD) robust anomaly score: 0.6745 * |x - median| / MAD."""
        clean = np.nan_to_num(signal_values)
        med = np.median(clean)
        mad = np.median(np.abs(clean - med))
        if mad < 1e-9:
            mad = 1.4826 * np.std(clean)  # fallback
        if mad < 1e-9:
            scores = np.zeros(len(clean))
        else:
            scores = 0.6745 * np.abs(clean - med) / mad
        is_anomaly = scores > threshold
        return is_anomaly, scores

    @staticmethod
    def detect_isolation_forest(signal_values: np.ndarray, contamination: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Isolation Forest on signal and its local derivative."""
        clean = np.nan_to_num(signal_values)
        diff = np.diff(clean, prepend=clean[0])
        X = np.column_stack([clean, diff])
        
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(X)
        # raw decision function: negative means outlier
        scores = -iso.decision_function(X)
        is_anomaly = preds == -1
        return is_anomaly, scores

    @staticmethod
    def detect_one_class_svm(signal_values: np.ndarray, nu: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """One-Class SVM with RBF kernel."""
        clean = np.nan_to_num(signal_values)
        diff = np.diff(clean, prepend=clean[0])
        X = np.column_stack([clean, diff])
        
        oc_svm = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
        preds = oc_svm.fit_predict(X)
        scores = -oc_svm.score_samples(X)
        is_anomaly = preds == -1
        return is_anomaly, scores


def plot_anomalies_waveform(
    time_axis: np.ndarray,
    signal_values: np.ndarray,
    is_anomaly: np.ndarray,
    x_label: str = "Time",
    method_name: str = "Anomaly Detection"
) -> go.Figure:
    """Plots signal waveform with detected anomaly points prominently marked."""
    fig = go.Figure()

    # Visual subsample for raw line if long
    n_pts = len(signal_values)
    stride = max(1, n_pts // 15000)
    t_plot = time_axis[::stride]
    s_plot = signal_values[::stride]

    # Baseline waveform
    fig.add_trace(go.Scatter(
        x=t_plot,
        y=s_plot,
        mode='lines',
        name='Nominal Signal',
        line=dict(color='#0284c7', width=1.4),
        opacity=0.85
    ))

    # Anomalies markers
    anomaly_indices = np.where(is_anomaly)[0]
    if len(anomaly_indices) > 0:
        fig.add_trace(go.Scatter(
            x=time_axis[anomaly_indices],
            y=signal_values[anomaly_indices],
            mode='markers',
            name=f'Detected Anomalies ({len(anomaly_indices)})',
            marker=dict(
                color='#ef4444',
                size=8,
                symbol='diamond-open-dot',
                line=dict(color='#dc2626', width=2)
            ),
            hovertemplate=f"<b>Anomaly Detected</b><br>{x_label}: %{{x:.4f}}<br>Amplitude: %{{y:.4f}}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text=f"Signal Waveform with Highlighted Anomalies ({method_name})", font=dict(size=15, color="#0f172a")),
        xaxis=dict(title=x_label, gridcolor="#f1f5f9"),
        yaxis=dict(title="Amplitude", gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    return fig


def plot_anomaly_scores(
    time_axis: np.ndarray,
    scores: np.ndarray,
    is_anomaly: np.ndarray,
    x_label: str = "Time"
) -> go.Figure:
    """Plots anomaly score sequence over time."""
    fig = go.Figure()

    stride = max(1, len(scores) // 15000)
    fig.add_trace(go.Scatter(
        x=time_axis[::stride],
        y=scores[::stride],
        mode='lines',
        name='Anomaly Score',
        line=dict(color='#f59e0b', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(245, 158, 11, 0.1)'
    ))

    anomaly_indices = np.where(is_anomaly)[0]
    if len(anomaly_indices) > 0:
        fig.add_trace(go.Scatter(
            x=time_axis[anomaly_indices],
            y=scores[anomaly_indices],
            mode='markers',
            name='Exceeded Threshold',
            marker=dict(color='#ef4444', size=6, symbol='circle')
        ))

    fig.update_layout(
        title=dict(text="Continuous Anomaly Score Trajectory", font=dict(size=15, color="#0f172a")),
        xaxis=dict(title=x_label, gridcolor="#f1f5f9"),
        yaxis=dict(title="Anomaly Score", gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=340
    )
    return fig

