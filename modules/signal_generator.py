"""
QUANTUM EYE — Built-in Scientific Signal Generator
Generates Periodic, Stochastic, and Deterministic Chaotic signals.
Supports multi-signal composition, noise injection by SNR (dB), and CSV export.
"""
import numpy as np
import pandas as pd
from scipy import signal
from typing import Dict, Any, Tuple, Optional

class SignalGenerator:
    """Scientific laboratory signal generator."""

    # ----------------- Periodic Signals -----------------
    @staticmethod
    def generate_sine(n_samples: int = 1000, fs: float = 100.0, freq: float = 5.0, amplitude: float = 1.0, phase: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        t = np.arange(n_samples) / fs
        s = amplitude * np.sin(2 * np.pi * freq * t + phase)
        return t, s

    @staticmethod
    def generate_square(n_samples: int = 1000, fs: float = 100.0, freq: float = 5.0, amplitude: float = 1.0, duty: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        t = np.arange(n_samples) / fs
        s = amplitude * signal.square(2 * np.pi * freq * t, duty=duty)
        return t, s

    @staticmethod
    def generate_sawtooth(n_samples: int = 1000, fs: float = 100.0, freq: float = 5.0, amplitude: float = 1.0, width: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        t = np.arange(n_samples) / fs
        s = amplitude * signal.sawtooth(2 * np.pi * freq * t, width=width)
        return t, s

    @staticmethod
    def generate_composite(n_samples: int = 1000, fs: float = 100.0, base_freq: float = 2.0, harmonics: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray]:
        t = np.arange(n_samples) / fs
        if harmonics is None:
            harmonics = [(1, 1.0), (3, 0.4), (5, 0.2)]  # (harmonic_multiplier, amplitude)
        s = np.zeros(n_samples)
        for mult, amp in harmonics:
            s += amp * np.sin(2 * np.pi * (base_freq * mult) * t)
        return t, s

    # ----------------- Stochastic Signals -----------------
    @staticmethod
    def generate_gaussian_noise(n_samples: int = 1000, fs: float = 100.0, mean: float = 0.0, std: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        t = np.arange(n_samples) / fs
        s = np.random.normal(mean, std, n_samples)
        return t, s

    @staticmethod
    def generate_uniform_noise(n_samples: int = 1000, fs: float = 100.0, low: float = -1.0, high: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        t = np.arange(n_samples) / fs
        s = np.random.uniform(low, high, n_samples)
        return t, s

    @staticmethod
    def generate_colored_noise(n_samples: int = 1000, fs: float = 100.0, color: str = "pink") -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates 1/f^alpha noise using spectral shaping in frequency domain.
        color: 'pink' (alpha=1.0) or 'brown'/'red' (alpha=2.0)
        """
        t = np.arange(n_samples) / fs
        white = np.random.randn(n_samples)
        fft_white = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples, d=1.0/fs)
        freqs[0] = freqs[1] if len(freqs) > 1 else 1.0  # avoid divide by zero

        alpha = 1.0 if color == "pink" else 2.0
        scale = 1.0 / (freqs ** (alpha / 2.0))
        shaped_fft = fft_white * scale
        shaped_fft[0] = 0.0
        s = np.fft.irfft(shaped_fft, n=n_samples)
        # Normalize to unit standard deviation
        s = s / (np.std(s) + 1e-9)
        return t, s

    # ----------------- Deterministic Chaotic Signals -----------------
    @staticmethod
    def generate_lorenz(n_samples: int = 2000, dt: float = 0.01, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0, variable: str = "x") -> Tuple[np.ndarray, np.ndarray]:
        """Runge-Kutta 4th order integration of the Lorenz 1963 system."""
        def lorenz_deriv(state):
            x, y, z = state
            return np.array([
                sigma * (y - x),
                x * (rho - z) - y,
                x * y - beta * z
            ])

        state = np.array([1.0, 1.0, 1.0])
        # Burn-in transient phase (1000 steps)
        for _ in range(1000):
            k1 = lorenz_deriv(state)
            k2 = lorenz_deriv(state + 0.5 * dt * k1)
            k3 = lorenz_deriv(state + 0.5 * dt * k2)
            k4 = lorenz_deriv(state + dt * k3)
            state += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        trajectory = np.empty((n_samples, 3))
        for i in range(n_samples):
            k1 = lorenz_deriv(state)
            k2 = lorenz_deriv(state + 0.5 * dt * k1)
            k3 = lorenz_deriv(state + 0.5 * dt * k2)
            k4 = lorenz_deriv(state + dt * k3)
            state += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            trajectory[i] = state

        t = np.arange(n_samples) * dt
        var_idx = {"x": 0, "y": 1, "z": 2}.get(variable.lower(), 0)
        return t, trajectory[:, var_idx]

    @staticmethod
    def generate_logistic_map(n_samples: int = 1000, r: float = 3.9, x0: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Logistic Map: x_{n+1} = r * x_n * (1 - x_n)."""
        x = x0
        # Burn-in
        for _ in range(200):
            x = r * x * (1.0 - x)

        s = np.empty(n_samples)
        for i in range(n_samples):
            x = r * x * (1.0 - x)
            s[i] = x
        t = np.arange(n_samples)
        return t, s

    @staticmethod
    def generate_rossler(n_samples: int = 2000, dt: float = 0.02, a: float = 0.2, b: float = 0.2, c: float = 5.7, variable: str = "x") -> Tuple[np.ndarray, np.ndarray]:
        """Runge-Kutta 4th order integration of the Rössler attractor."""
        def rossler_deriv(state):
            x, y, z = state
            return np.array([
                -y - z,
                x + a * y,
                b + z * (x - c)
            ])

        state = np.array([0.1, 0.1, 0.1])
        # Burn-in
        for _ in range(1000):
            k1 = rossler_deriv(state)
            k2 = rossler_deriv(state + 0.5 * dt * k1)
            k3 = rossler_deriv(state + 0.5 * dt * k2)
            k4 = rossler_deriv(state + dt * k3)
            state += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        trajectory = np.empty((n_samples, 3))
        for i in range(n_samples):
            k1 = rossler_deriv(state)
            k2 = rossler_deriv(state + 0.5 * dt * k1)
            k3 = rossler_deriv(state + 0.5 * dt * k2)
            k4 = rossler_deriv(state + dt * k3)
            state += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            trajectory[i] = state

        t = np.arange(n_samples) * dt
        var_idx = {"x": 0, "y": 1, "z": 2}.get(variable.lower(), 0)
        return t, trajectory[:, var_idx]

    # ----------------- Signal Mixer & Noise Addition -----------------
    @staticmethod
    def inject_noise_snr(signal_values: np.ndarray, snr_db: Optional[float] = None, noise_std: Optional[float] = None) -> np.ndarray:
        """
        Adds Gaussian white noise specified by either:
        - SNR in decibels: SNR_dB = 10 * log10(P_signal / P_noise)
        - Absolute noise standard deviation (noise_std)
        """
        p_signal = np.mean(signal_values ** 2)
        if snr_db is not None:
            if np.isinf(snr_db) or snr_db > 100:
                return signal_values.copy()
            # P_noise = P_signal / (10^(SNR_db / 10))
            p_noise = p_signal / (10.0 ** (snr_db / 10.0))
            sigma_noise = np.sqrt(max(0, p_noise))
        elif noise_std is not None:
            sigma_noise = float(noise_std)
        else:
            return signal_values.copy()

        noise = np.random.normal(0, sigma_noise, len(signal_values))
        return signal_values + noise

    @staticmethod
    def export_to_dataframe(time_axis: np.ndarray, signal_values: np.ndarray, signal_name: str = "generated_signal") -> pd.DataFrame:
        """Wraps time and signal in a clean DataFrame."""
        return pd.DataFrame({
            "time": np.round(time_axis, 4),
            signal_name: np.round(signal_values, 5)
        })

