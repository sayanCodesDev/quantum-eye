"""
QUANTUM EYE — Signal Generator Laboratory
Generates Periodic, Stochastic, and Chaotic signals with interactive mixing,
SNR noise injection, CSV export, and instant ingestion into the analysis pipeline.
"""
import streamlit as st
import numpy as np
import pandas as pd
from modules.signal_generator import SignalGenerator
from modules.signal_analysis import plot_waveform
from modules.frequency_analysis import compute_welch_psd, plot_psd_spectrum
from modules.state_manager import set_active_dataset

def render():
    st.title("🧪 Built-in Scientific Signal Generator")
    st.markdown("Generate synthetic, stochastic, and chaotic waveforms. Combine multi-modal harmonics, inject noise with calibrated SNR (dB), and ingest directly into Quantum Eye.")

    tab_periodic, tab_stoch, tab_chaos, tab_mixer = st.tabs([
        "🌊 Periodic Signals", "🎲 Stochastic Noise", "🌀 Chaotic Attractors", "🎛️ Signal Mixer & Noise Injection"
    ])

    fs_default = 100.0
    n_samples_default = 1000

    # 1. Periodic Signals
    with tab_periodic:
        st.subheader("Periodic Waveform Synthesis")
        c1, c2, c3 = st.columns(3)
        with c1:
            p_type = st.selectbox("Waveform Type", ["Sine Wave", "Square Wave", "Sawtooth Wave", "Composite Multi-Harmonic"])
        with c2:
            p_freq = st.slider("Fundamental Frequency (Hz)", 0.5, 30.0, 5.0, 0.5)
        with c3:
            p_amp = st.slider("Amplitude", 0.1, 5.0, 1.0, 0.1)

        c_s1, c_s2 = st.columns(2)
        with c_s1:
            p_samples = st.slider("Sample Count", 200, 5000, 1000, 100, key="p_s")
        with c_s2:
            p_fs = st.slider("Sampling Rate (Hz)", 20.0, 500.0, 100.0, 10.0, key="p_fs")

        if p_type == "Sine Wave":
            t, s = SignalGenerator.generate_sine(n_samples=p_samples, fs=p_fs, freq=p_freq, amplitude=p_amp)
        elif p_type == "Square Wave":
            t, s = SignalGenerator.generate_square(n_samples=p_samples, fs=p_fs, freq=p_freq, amplitude=p_amp)
        elif p_type == "Sawtooth Wave":
            t, s = SignalGenerator.generate_sawtooth(n_samples=p_samples, fs=p_fs, freq=p_freq, amplitude=p_amp)
        else:
            t, s = SignalGenerator.generate_composite(n_samples=p_samples, fs=p_fs, base_freq=p_freq)

        _display_generated_signal(t, s, name=f"periodic_{p_type.lower().replace(' ', '_')}", fs=p_fs)

    # 2. Stochastic Noise
    with tab_stoch:
        st.subheader("Stochastic Process Synthesis")
        c_st1, c_st2, c_st3 = st.columns(3)
        with c_st1:
            stoch_type = st.selectbox("Stochastic Process", ["Gaussian White Noise", "Uniform Noise", "Pink Noise (1/f)", "Brown Noise (1/f²)"])
        with c_st2:
            stoch_samples = st.slider("Sample Count", 200, 5000, 1000, 100, key="stoch_s")
        with c_st3:
            stoch_fs = st.slider("Sampling Rate (Hz)", 20.0, 500.0, 100.0, 10.0, key="stoch_fs")

        if stoch_type == "Gaussian White Noise":
            t, s = SignalGenerator.generate_gaussian_noise(n_samples=stoch_samples, fs=stoch_fs)
        elif stoch_type == "Uniform Noise":
            t, s = SignalGenerator.generate_uniform_noise(n_samples=stoch_samples, fs=stoch_fs)
        elif stoch_type.startswith("Pink"):
            t, s = SignalGenerator.generate_colored_noise(n_samples=stoch_samples, fs=stoch_fs, color="pink")
        else:
            t, s = SignalGenerator.generate_colored_noise(n_samples=stoch_samples, fs=stoch_fs, color="brown")

        _display_generated_signal(t, s, name=f"stochastic_{stoch_type.lower().split()[0]}", fs=stoch_fs)

    # 3. Deterministic Chaos
    with tab_chaos:
        st.subheader("Deterministic Chaotic System Integration")
        c_ch1, c_ch2, c_ch3 = st.columns(3)
        with c_ch1:
            chaos_type = st.selectbox("Chaotic System", ["Lorenz Attractor (1963)", "Logistic Map", "Rössler Attractor"])
        with c_ch2:
            ch_samples = st.slider("Sample Length", 300, 5000, 1500, 100, key="ch_s")
        with c_ch3:
            ch_coord = st.selectbox("State Variable", ["x", "y", "z"])

        if chaos_type.startswith("Lorenz"):
            c_p1, c_p2, c_p3 = st.columns(3)
            with c_p1:
                sigma = st.slider("Prandtl (σ)", 5.0, 20.0, 10.0, 0.5)
            with c_p2:
                rho = st.slider("Rayleigh (ρ)", 15.0, 40.0, 28.0, 0.5)
            with c_p3:
                beta = st.slider("Aspect (β)", 1.0, 5.0, 8.0/3.0, 0.1)
            t, s = SignalGenerator.generate_lorenz(n_samples=ch_samples, sigma=sigma, rho=rho, beta=beta, variable=ch_coord)
            fs_val = 100.0
        elif chaos_type == "Logistic Map":
            r_param = st.slider("Growth Parameter (r)", 2.8, 4.0, 3.9, 0.01)
            t, s = SignalGenerator.generate_logistic_map(n_samples=ch_samples, r=r_param)
            fs_val = None
        else:
            t, s = SignalGenerator.generate_rossler(n_samples=ch_samples, variable=ch_coord)
            fs_val = 50.0

        _display_generated_signal(t, s, name=f"chaotic_{chaos_type.lower().split()[0]}_{ch_coord}", fs=fs_val)

    # 4. Signal Mixer & Controlled SNR Injection
    with tab_mixer:
        st.subheader("Complex Signal Mixer & Noise Degradation Experiment")
        st.markdown("""
        Create hybrid signals (e.g., **Lorenz Attractor + Multi-harmonic Sine + Pink Noise**)
        and observe how entropy, spectral distribution, and ML classification shift.
        """)

        c_mix1, c_mix2 = st.columns(2)
        with c_mix1:
            base_choice = st.selectbox("Primary Signal Carrier", ["Lorenz Attractor (Chaos)", "Sine Wave (Periodic)", "Composite Wave"])
            weight_base = st.slider("Primary Weight", 0.1, 2.0, 1.0, 0.1)
        with c_mix2:
            mod_choice = st.selectbox("Additive Modulator / Harmonic", ["Gaussian Noise", "Pink Noise", "Sine Wave (High Freq)", "None"])
            weight_mod = st.slider("Modulator Weight", 0.0, 2.0, 0.3, 0.1)

        snr_inject = st.selectbox("Injected Noise Level (SNR in dB)", [None, 30.0, 20.0, 15.0, 10.0, 5.0, 0.0],
                                  format_func=lambda x: "No Additional Gaussian Noise" if x is None else f"{x} dB SNR")

        # Synthesize mixed signal
        n_m = 1200
        t_mix = np.arange(n_m) * 0.01
        
        if "Lorenz" in base_choice:
            _, s_base = SignalGenerator.generate_lorenz(n_samples=n_m)
        elif "Sine" in base_choice:
            _, s_base = SignalGenerator.generate_sine(n_samples=n_m, fs=100.0, freq=3.0)
        else:
            _, s_base = SignalGenerator.generate_composite(n_samples=n_m, fs=100.0, base_freq=2.0)

        s_mixed = weight_base * s_base

        if mod_choice == "Gaussian Noise":
            _, s_mod = SignalGenerator.generate_gaussian_noise(n_samples=n_m)
            s_mixed += weight_mod * s_mod
        elif mod_choice == "Pink Noise":
            _, s_mod = SignalGenerator.generate_colored_noise(n_samples=n_m, color="pink")
            s_mixed += weight_mod * s_mod
        elif "Sine" in mod_choice:
            _, s_mod = SignalGenerator.generate_sine(n_samples=n_m, fs=100.0, freq=25.0)
            s_mixed += weight_mod * s_mod

        if snr_inject is not None:
            s_mixed = SignalGenerator.inject_noise_snr(s_mixed, snr_db=snr_inject)

        _display_generated_signal(t_mix, s_mixed, name="hybrid_mixed_signal", fs=100.0)


def _display_generated_signal(t: np.ndarray, s: np.ndarray, name: str, fs: float):
    """Helper to render preview and action buttons."""
    c_p1, c_p2 = st.columns([1.5, 1])
    with c_p1:
        fig_w = plot_waveform(t, s, title=f"Waveform Preview: {name}", signal_name=name, x_label=f"Time ({'s' if fs else 'Index'})")
        st.plotly_chart(fig_w, use_container_width=True)
    with c_p2:
        freqs, psd, unit = compute_welch_psd(s, fs=fs)
        fig_psd = plot_psd_spectrum(freqs, psd, freq_unit=unit)
        st.plotly_chart(fig_psd, use_container_width=True)

    df_gen = pd.DataFrame({"time": t, name: np.round(s, 5)})
    
    col_act1, col_act2, _ = st.columns([1, 1, 2])
    with col_act1:
        if st.button(f"📥 Ingest into Laboratory ({name})", type="primary", use_container_width=True):
            set_active_dataset(df_gen, name=f"Synthetic: {name}", signal_col=name, time_col="time", fs=fs)
            st.success(f"Ingested '{name}' as the active laboratory signal!")
            st.rerun()

    with col_act2:
        csv_data = df_gen.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download as CSV",
            data=csv_data,
            file_name=f"{name}.csv",
            mime="text/csv",
            use_container_width=True
        )

