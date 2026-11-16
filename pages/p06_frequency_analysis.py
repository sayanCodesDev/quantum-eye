"""
QUANTUM EYE — Page 6: Frequency-Domain Spectral Analysis
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from modules.state_manager import get_current_signal
from modules.frequency_analysis import (
    compute_fft_spectrum, compute_welch_psd, compute_spectral_metrics,
    plot_fft_spectrum, plot_psd_spectrum
)

def render():
    st.title("⚡ Frequency-Domain Spectral Analysis")
    st.markdown("Deconstruct continuous signals into harmonic components using Fast Fourier Transform (FFT) and Welch's Power Spectral Density periodogram.")

    time_axis, signal_values, sig_name, fs = get_current_signal()
    if signal_values is None or len(signal_values) == 0:
        st.warning("No signal loaded. Please select or upload a dataset.")
        return

    # Guardrail check for Sampling Frequency
    if fs is None:
        st.markdown("""
        <div class="qe-callout qe-callout-warning">
            <b>Sampling Frequency Unknown:</b> Using normalized frequency (cycles per sample).
            Frequencies will not be falsely labeled in Hertz (Hz) to preserve physical accuracy.
            You can set a sampling frequency in <b>Upload Dataset</b>.
        </div>
        """, unsafe_allow_html=True)

    # Compute FFT and Welch PSD
    fft_freqs, fft_amps, freq_unit = compute_fft_spectrum(signal_values, fs=fs)
    psd_freqs, psd_vals, _ = compute_welch_psd(signal_values, fs=fs)
    spec_metrics = compute_spectral_metrics(psd_freqs, psd_vals, freq_unit=freq_unit)
    
    st.session_state.setdefault("cached_results", {})["spectral_metrics"] = spec_metrics

    # Spectral Metrics Grid
    st.subheader(f"🎛️ Spectral Measurements: `{sig_name}`")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dominant Frequency", f"{spec_metrics.get('dominant_frequency', 0.0):.3f} {freq_unit}")
    c2.metric("Spectral Centroid", f"{spec_metrics.get('spectral_centroid', 0.0):.3f} {freq_unit}")
    c3.metric("Spectral Spread", f"{spec_metrics.get('spectral_spread', 0.0):.3f}")
    c4.metric("Spectral Entropy", f"{spec_metrics.get('spectral_entropy', 0.0):.4f} bits", delta=f"Normalized: {spec_metrics.get('normalized_spectral_entropy', 0.0):.3f}")

    st.markdown("---")

    # Interactive Spectral Visualizations
    tab_fft, tab_psd, tab_peaks = st.tabs([
        "📊 FFT Amplitude Spectrum", "📈 Welch Power Spectral Density", "🎯 Dominant Harmonic Peaks"
    ])

    with tab_fft:
        st.subheader("Fast Fourier Transform (FFT) Amplitude Spectrum")
        c_p, _ = st.columns([1, 3])
        with c_p:
            n_peaks_fft = st.slider("Highlight Top Peaks", min_value=1, max_value=8, value=3)
        fig_fft = plot_fft_spectrum(fft_freqs, fft_amps, freq_unit=freq_unit, top_n_peaks=n_peaks_fft)
        st.plotly_chart(fig_fft, use_container_width=True)

    with tab_psd:
        st.subheader("Welch's Power Spectral Density (PSD)")
        c_scale, _ = st.columns([1, 3])
        with c_scale:
            use_log = st.checkbox("Logarithmic Power Scale (dB)", value=True)
        fig_psd = plot_psd_spectrum(psd_freqs, psd_vals, freq_unit=freq_unit, log_scale=use_log)
        st.plotly_chart(fig_psd, use_container_width=True)

    with tab_peaks:
        st.subheader("Harmonic Peak Table & Energy Distribution")
        # Rank top 10 frequencies by power
        peak_idx = np.argsort(psd_vals)[-10:][::-1]
        peaks_df = pd.DataFrame({
            f"Frequency ({freq_unit})": np.round(psd_freqs[peak_idx], 4),
            "Power Density": np.round(psd_vals[peak_idx], 6),
            "Power (dB)": np.round(10 * np.log10(np.maximum(psd_vals[peak_idx], 1e-12)), 2),
            "Relative Power Share (%)": np.round((psd_vals[peak_idx] / np.sum(psd_vals)) * 100, 2)
        })
        st.dataframe(peaks_df, use_container_width=True)
        st.caption("Top 10 spectral frequencies ordered by power density.")

