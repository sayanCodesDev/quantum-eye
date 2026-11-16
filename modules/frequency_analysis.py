"""
QUANTUM EYE — Frequency-Domain Signal Analysis Engine
Computes FFT, Welch's Power Spectral Density, Spectral Centroid,
Spectral Entropy, and generates interactive Plotly spectral charts.
"""
import numpy as np
from scipy import signal, fft
import plotly.graph_objects as go
from typing import Dict, Any, Tuple, Optional

def compute_fft_spectrum(signal_values: np.ndarray, fs: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Computes one-sided FFT amplitude spectrum.
    If fs is None, normalized frequency (cycles/sample) is used.
    Returns: (freqs, amplitudes, freq_unit)
    """
    clean_vals = signal_values[~np.isnan(signal_values)]
    n = len(clean_vals)
    if n == 0:
        return np.array([]), np.array([]), "Cycles/Sample"

    # Subtract mean to remove zero-frequency DC spike
    detrended = clean_vals - np.mean(clean_vals)
    
    # Apply Hanning window to suppress spectral leakage
    window = np.hanning(n)
    windowed = detrended * window
    
    # Compute FFT
    fft_vals = fft.rfft(windowed)
    amplitudes = (2.0 / n) * np.abs(fft_vals)
    
    if fs is not None and fs > 0:
        freqs = fft.rfftfreq(n, d=1.0/fs)
        freq_unit = "Hz"
    else:
        freqs = fft.rfftfreq(n, d=1.0) # normalized frequency [0, 0.5] cycles/sample
        freq_unit = "Normalized Freq (cycles/sample)"

    return freqs, amplitudes, freq_unit


def compute_welch_psd(signal_values: np.ndarray, fs: Optional[float] = None, nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Computes Power Spectral Density using Welch's averaged periodogram method.
    """
    clean_vals = signal_values[~np.isnan(signal_values)]
    n = len(clean_vals)
    if n == 0:
        return np.array([]), np.array([]), "Cycles/Sample"

    detrended = clean_vals - np.mean(clean_vals)
    effective_fs = fs if (fs is not None and fs > 0) else 1.0
    freq_unit = "Hz" if (fs is not None and fs > 0) else "Normalized Freq (cycles/sample)"
    
    if nperseg is None:
        nperseg = min(256, n)
    elif nperseg > n:
        nperseg = n

    freqs, psd = signal.welch(detrended, fs=effective_fs, window='hann', nperseg=nperseg, noverlap=nperseg//2)
    return freqs, psd, freq_unit


def compute_spectral_metrics(freqs: np.ndarray, psd: np.ndarray, freq_unit: str = "Hz") -> Dict[str, Any]:
    """
    Calculates dominant frequency, spectral centroid, spectral spread, and spectral entropy.
    """
    if len(freqs) == 0 or len(psd) == 0:
        return {}

    # Dominant frequency
    max_idx = np.argmax(psd)
    dom_freq = float(freqs[max_idx])
    peak_power = float(psd[max_idx])

    # Spectral centroid: sum(f * P(f)) / sum(P(f))
    total_power = np.sum(psd)
    if total_power > 1e-12:
        spectral_centroid = float(np.sum(freqs * psd) / total_power)
        # Spectral spread (standard deviation around centroid)
        spectral_spread = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / total_power))
        # Spectral entropy: -sum(p * log2(p)) where p = P(f) / sum(P(f))
        p = psd / total_power
        p = p[p > 0]
        spectral_entropy = float(-np.sum(p * np.log2(p)))
        # Normalized spectral entropy between [0, 1]
        max_entropy = np.log2(len(psd)) if len(psd) > 1 else 1.0
        normalized_spectral_entropy = float(spectral_entropy / max_entropy) if max_entropy > 0 else 0.0
    else:
        spectral_centroid = 0.0
        spectral_spread = 0.0
        spectral_entropy = 0.0
        normalized_spectral_entropy = 0.0

    return {
        "dominant_frequency": dom_freq,
        "peak_power": peak_power,
        "spectral_centroid": spectral_centroid,
        "spectral_spread": spectral_spread,
        "spectral_entropy": spectral_entropy,
        "normalized_spectral_entropy": normalized_spectral_entropy,
        "freq_unit": freq_unit,
        "total_power": float(total_power)
    }


def plot_fft_spectrum(freqs: np.ndarray, amplitudes: np.ndarray, freq_unit: str = "Hz", top_n_peaks: int = 3) -> go.Figure:
    """Plots interactive FFT Amplitude Spectrum with dominant peak annotations."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=freqs,
        y=amplitudes,
        mode='lines',
        name='FFT Amplitude',
        line=dict(color='#0284c7', width=1.6),
        hovertemplate=f"<b>Freq</b>: %{{x:.3f}} {freq_unit}<br><b>Amplitude</b>: %{{y:.4f}}<extra></extra>"
    ))

    # Mark top peaks
    if len(amplitudes) > 5:
        peak_indices = np.argsort(amplitudes)[-top_n_peaks:][::-1]
        fig.add_trace(go.Scatter(
            x=freqs[peak_indices],
            y=amplitudes[peak_indices],
            mode='markers+text',
            name='Dominant Peaks',
            text=[f"{freqs[i]:.2f} {freq_unit}" for i in peak_indices],
            textposition='top right',
            marker=dict(color='#ef4444', size=8, symbol='diamond')
        ))

    fig.update_layout(
        title=dict(text="Fast Fourier Transform (FFT) Amplitude Spectrum", font=dict(size=15, color="#0f172a")),
        xaxis=dict(title=f"Frequency ({freq_unit})", gridcolor="#f1f5f9"),
        yaxis=dict(title="Amplitude", gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=50, b=40),
        height=380
    )
    return fig


def plot_psd_spectrum(freqs: np.ndarray, psd: np.ndarray, freq_unit: str = "Hz", log_scale: bool = True) -> go.Figure:
    """Plots Welch's Power Spectral Density."""
    fig = go.Figure()

    y_vals = 10 * np.log10(np.maximum(psd, 1e-12)) if log_scale else psd
    y_title = "Power/Freq (dB/Hz)" if log_scale else "Power/Freq"

    fig.add_trace(go.Scatter(
        x=freqs,
        y=y_vals,
        mode='lines',
        name='Welch PSD',
        line=dict(color='#0d9488', width=1.8),
        fill='tozeroy',
        fillcolor='rgba(13, 148, 136, 0.1)',
        hovertemplate=f"<b>Freq</b>: %{{x:.3f}} {freq_unit}<br><b>{y_title}</b>: %{{y:.2f}}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text=f"Power Spectral Density (Welch Periodogram - {y_title})", font=dict(size=15, color="#0f172a")),
        xaxis=dict(title=f"Frequency ({freq_unit})", gridcolor="#f1f5f9"),
        yaxis=dict(title=y_title, gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=50, b=40),
        height=380
    )
    return fig

