"""
QUANTUM EYE — Time-Domain Signal Analysis Engine
Calculates time-domain metrics, rolling statistics, distribution moments,
and generates interactive Plotly visualizations.
"""
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from typing import Dict, Any, Tuple, Optional

def compute_time_domain_metrics(signal_values: np.ndarray) -> Dict[str, float]:
    """
    Computes rigorous time-domain statistical metrics on a 1D signal.
    """
    clean_vals = signal_values[~np.isnan(signal_values)]
    n = len(clean_vals)
    if n == 0:
        return {}

    mean = float(np.mean(clean_vals))
    median = float(np.median(clean_vals))
    variance = float(np.var(clean_vals))
    std = float(np.std(clean_vals))
    s_min = float(np.min(clean_vals))
    s_max = float(np.max(clean_vals))
    s_range = s_max - s_min
    skewness = float(stats.skew(clean_vals)) if std > 1e-9 else 0.0
    kurtosis = float(stats.kurtosis(clean_vals)) if std > 1e-9 else 0.0 # Fisher excess kurtosis
    rms = float(np.sqrt(np.mean(clean_vals ** 2)))
    p2p = s_range
    crest_factor = float(np.max(np.abs(clean_vals)) / rms) if rms > 1e-9 else 0.0
    zero_crossings = int(np.sum(np.diff(np.sign(clean_vals - mean) != 0)))
    
    return {
        "Mean": mean,
        "Median": median,
        "Variance": variance,
        "Std Dev": std,
        "Min": s_min,
        "Max": s_max,
        "Range": s_range,
        "Skewness": skewness,
        "Excess Kurtosis": kurtosis,
        "RMS": rms,
        "Peak-to-Peak": p2p,
        "Crest Factor": crest_factor,
        "Zero Crossings": zero_crossings,
        "Sample Count": n
    }


def plot_waveform(
    time_axis: np.ndarray, 
    signal_values: np.ndarray, 
    title: str = "Signal Waveform", 
    signal_name: str = "Signal",
    color: str = "#0284c7",
    x_label: str = "Time",
    y_label: str = "Amplitude"
) -> go.Figure:
    """Generates clean interactive Plotly waveform."""
    fig = go.Figure()
    
    # Downsample visually if points > 15000 to prevent WebGL browser lag
    n_pts = len(signal_values)
    stride = max(1, n_pts // 15000)
    t_plot = time_axis[::stride]
    s_plot = signal_values[::stride]

    fig.add_trace(go.Scatter(
        x=t_plot,
        y=s_plot,
        mode='lines',
        name=signal_name,
        line=dict(color=color, width=1.5),
        hovertemplate=f"<b>{x_label}</b>: %{{x:.4f}}<br><b>{y_label}</b>: %{{y:.4f}}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(family="Inter, sans-serif", size=15, color="#0f172a")),
        xaxis=dict(title=x_label, gridcolor="#f1f5f9", zeroline=False),
        yaxis=dict(title=y_label, gridcolor="#f1f5f9", zeroline=False),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="x unified",
        height=380
    )
    return fig


def plot_comparison_waveform(
    time_axis: np.ndarray,
    orig_signal: np.ndarray,
    proc_signal: np.ndarray,
    orig_name: str = "Original",
    proc_name: str = "Processed",
    x_label: str = "Time",
    y_label: str = "Amplitude"
) -> go.Figure:
    """Displays original vs processed signal overlay."""
    fig = go.Figure()
    
    n_pts = len(orig_signal)
    stride = max(1, n_pts // 15000)
    t_plot = time_axis[::stride]
    
    fig.add_trace(go.Scatter(
        x=t_plot,
        y=orig_signal[::stride],
        mode='lines',
        name=orig_name,
        line=dict(color="#94a3b8", width=1.2, dash='dot'),
        opacity=0.75
    ))

    fig.add_trace(go.Scatter(
        x=t_plot,
        y=proc_signal[::stride],
        mode='lines',
        name=proc_name,
        line=dict(color="#0284c7", width=1.8),
        opacity=0.95
    ))

    fig.update_layout(
        title=dict(text="Original vs. Processed Signal Waveform", font=dict(size=15, color="#0f172a")),
        xaxis=dict(title=x_label, gridcolor="#f1f5f9"),
        yaxis=dict(title=y_label, gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    return fig


def plot_rolling_statistics(
    time_axis: np.ndarray,
    signal_values: np.ndarray,
    window_size: int = 50,
    x_label: str = "Time"
) -> go.Figure:
    """Computes and plots rolling mean and confidence envelope (+/- 2 std)."""
    s = pd.Series(signal_values)
    rolling_m = s.rolling(window=window_size, min_periods=1, center=True).mean().values
    rolling_s = s.rolling(window=window_size, min_periods=1, center=True).std().fillna(0).values
    
    upper = rolling_m + 2 * rolling_s
    lower = rolling_m - 2 * rolling_s
    
    n_pts = len(signal_values)
    stride = max(1, n_pts // 15000)
    t_plot = time_axis[::stride]

    fig = go.Figure()
    
    # Uncertainty ribbon
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_plot, t_plot[::-1]]),
        y=np.concatenate([upper[::stride], lower[::stride][::-1]]),
        fill='toself',
        fillcolor='rgba(2, 132, 199, 0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='±2σ Rolling Band'
    ))

    # Raw signal
    fig.add_trace(go.Scatter(
        x=t_plot,
        y=signal_values[::stride],
        mode='lines',
        name='Signal',
        line=dict(color='#94a3b8', width=0.9),
        opacity=0.5
    ))

    # Rolling mean
    fig.add_trace(go.Scatter(
        x=t_plot,
        y=rolling_m[::stride],
        mode='lines',
        name=f'Rolling Mean (w={window_size})',
        line=dict(color='#0284c7', width=2.0)
    ))

    fig.update_layout(
        title=dict(text=f"Rolling Mean & Deviation Envelope (Window = {window_size})", font=dict(size=15, color="#0f172a")),
        xaxis=dict(title=x_label, gridcolor="#f1f5f9"),
        yaxis=dict(title="Amplitude", gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380
    )
    return fig


def plot_distribution(signal_values: np.ndarray, bins: int = 50) -> go.Figure:
    """Plots histogram with fitted normal probability density curve."""
    clean_vals = signal_values[~np.isnan(signal_values)]
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=clean_vals,
        nbinsx=bins,
        histnorm='probability density',
        name='Sample Density',
        marker=dict(color='#0284c7', line=dict(color='#ffffff', width=1)),
        opacity=0.75
    ))

    # Overlay Gaussian fit
    mu, std = np.mean(clean_vals), np.std(clean_vals)
    if std > 1e-9:
        x_norm = np.linspace(np.min(clean_vals), np.max(clean_vals), 200)
        p_norm = stats.norm.pdf(x_norm, mu, std)
        fig.add_trace(go.Scatter(
            x=x_norm,
            y=p_norm,
            mode='lines',
            name=f'Fitted Normal (μ={mu:.2f}, σ={std:.2f})',
            line=dict(color='#0d9488', width=2.2)
        ))

    fig.update_layout(
        title=dict(text="Amplitude Distribution & Gaussian Fit", font=dict(size=15, color="#0f172a")),
        xaxis=dict(title="Amplitude Value", gridcolor="#f1f5f9"),
        yaxis=dict(title="Probability Density", gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=360
    )
    return fig


def plot_boxplot(signal_values: np.ndarray, signal_name: str = "Signal") -> go.Figure:
    """Plots boxplot with quartile and outlier markings."""
    clean_vals = signal_values[~np.isnan(signal_values)]
    fig = go.Figure()

    fig.add_trace(go.Box(
        y=clean_vals,
        name=signal_name,
        boxpoints='outliers',
        jitter=0.3,
        pointpos=-1.8,
        marker=dict(color='#8b5cf6', size=4),
        line=dict(color='#0284c7', width=1.5),
        fillcolor='rgba(2, 132, 199, 0.2)'
    ))

    fig.update_layout(
        title=dict(text=f"Box Plot & Quantile Spread ({signal_name})", font=dict(size=15, color="#0f172a")),
        yaxis=dict(title="Amplitude Value", gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=50, b=40),
        height=360
    )
    return fig

