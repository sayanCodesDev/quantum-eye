"""
QUANTUM EYE — Automated Scientific Feature Extraction Engine
Extracts Statistical, Frequency, and Nonlinear features from 1D signals
or windowed series. Guarantees safe numeric handling without leakage.
"""
import numpy as np
import pandas as pd
from scipy import stats, fft, signal
from typing import Dict, Any, List, Optional
from modules.frequency_analysis import compute_spectral_metrics, compute_welch_psd
from modules.nonlinear_analysis import (
    compute_shannon_entropy,
    compute_permutation_entropy,
    compute_approximate_entropy,
    compute_sample_entropy
)

def extract_signal_features(signal_values: np.ndarray, fs: Optional[float] = None) -> Dict[str, float]:
    """
    Extracts a comprehensive feature vector (statistical, frequency, nonlinear) from a 1D signal.
    """
    clean = signal_values[~np.isnan(signal_values)]
    n = len(clean)
    if n == 0:
        return {}

    effective_fs = fs if (fs is not None and fs > 0) else 1.0

    # 1. Statistical Features
    mean_val = float(np.mean(clean))
    std_val = float(np.std(clean))
    var_val = float(np.var(clean))
    min_val = float(np.min(clean))
    max_val = float(np.max(clean))
    p2p_val = max_val - min_val
    skew_val = float(stats.skew(clean)) if std_val > 1e-9 else 0.0
    kurt_val = float(stats.kurtosis(clean)) if std_val > 1e-9 else 0.0
    rms_val = float(np.sqrt(np.mean(clean ** 2)))
    crest_val = float(np.max(np.abs(clean)) / rms_val) if rms_val > 1e-9 else 0.0
    zero_crossings = int(np.sum(np.diff(np.sign(clean - mean_val) != 0)))
    zcr = float(zero_crossings / n)

    # Autocorrelation lag-1
    if std_val > 1e-9 and n > 2:
        norm_s = clean - mean_val
        autocorr_1 = float(np.corrcoef(norm_s[:-1], norm_s[1:])[0, 1])
        if np.isnan(autocorr_1):
            autocorr_1 = 0.0
    else:
        autocorr_1 = 0.0

    # 2. Frequency Features
    freqs, psd, _ = compute_welch_psd(clean, fs=effective_fs)
    spec_metrics = compute_spectral_metrics(freqs, psd)

    dom_freq = spec_metrics.get("dominant_frequency", 0.0)
    spec_centroid = spec_metrics.get("spectral_centroid", 0.0)
    spec_entropy = spec_metrics.get("spectral_entropy", 0.0)
    norm_spec_entropy = spec_metrics.get("normalized_spectral_entropy", 0.0)
    
    # Band Powers (Low band: 0 - 0.25*Nyquist, High band: 0.25 - Nyquist)
    if len(freqs) > 2:
        max_f = freqs[-1]
        low_mask = freqs <= (0.25 * max_f)
        high_mask = freqs > (0.25 * max_f)
        low_band_power = float(np.sum(psd[low_mask]))
        high_band_power = float(np.sum(psd[high_mask]))
        band_power_ratio = float(low_band_power / (high_band_power + 1e-9))
    else:
        low_band_power, high_band_power, band_power_ratio = 0.0, 0.0, 0.0

    # 3. Nonlinear Features
    shannon_ent = compute_shannon_entropy(clean)
    _, norm_perm_ent = compute_permutation_entropy(clean, m=3, tau=1)
    
    # Compute ApEn and SampEn on a fast subsample if signal is long
    sub_for_ent = clean[::max(1, n // 500)]
    apen = compute_approximate_entropy(sub_for_ent, m=2)
    sampen = compute_sample_entropy(sub_for_ent, m=2)

    features = {
        # Statistical
        "stat_mean": mean_val,
        "stat_std": std_val,
        "stat_variance": var_val,
        "stat_min": min_val,
        "stat_max": max_val,
        "stat_peak_to_peak": p2p_val,
        "stat_skewness": skew_val,
        "stat_kurtosis": kurt_val,
        "stat_rms": rms_val,
        "stat_crest_factor": crest_val,
        "stat_zero_crossing_rate": zcr,
        "stat_autocorr_lag1": autocorr_1,
        
        # Frequency
        "freq_dominant": dom_freq,
        "freq_spectral_centroid": spec_centroid,
        "freq_spectral_entropy": spec_entropy,
        "freq_norm_spectral_entropy": norm_spec_entropy,
        "freq_low_band_power": low_band_power,
        "freq_high_band_power": high_band_power,
        "freq_band_power_ratio": band_power_ratio,
        
        # Nonlinear
        "nl_shannon_entropy": shannon_ent,
        "nl_perm_entropy": norm_perm_ent,
        "nl_approx_entropy": apen,
        "nl_sample_entropy": sampen
    }

    # Replace any potential inf/nan safely
    for k, v in features.items():
        if np.isnan(v) or np.isinf(v):
            features[k] = 0.0

    return features


def extract_windowed_feature_matrix(
    signal_values: np.ndarray, 
    window_size: int = 200, 
    step_size: int = 100, 
    fs: Optional[float] = None
) -> pd.DataFrame:
    """
    Extracts features across sliding temporal windows over a long signal.
    """
    clean = signal_values[~np.isnan(signal_values)]
    n = len(clean)
    if n < window_size:
        # Just extract single window
        f = extract_signal_features(clean, fs=fs)
        return pd.DataFrame([f])

    records = []
    window_indices = []
    
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        win = clean[start:end]
        f = extract_signal_features(win, fs=fs)
        records.append(f)
        window_indices.append(f"{start}:{end}")

    df_feats = pd.DataFrame(records)
    df_feats.index = window_indices
    df_feats.index.name = "window_range"
    return df_feats

