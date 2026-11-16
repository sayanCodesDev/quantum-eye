"""
Generate benchmark and sample datasets for Quantum Eye.
Includes datasets inspired by UCR/UEA Time Series Classification Archive:
- ECG200 Heartbeat sample (Normal vs Myocardial Infarction)
- Synthetic Control Charts (6 patterns)
- Lorenz Deterministic Chaos series
- Mackey-Glass Delayed Differential Equation series
"""
import numpy as np
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

# 1. ECG200 Heartbeat Synthetic Approximation
# Standard ECG waveform: P wave, QRS complex, T wave
np.random.seed(42)
t = np.linspace(0, 1.0, 100) # 100 sample points per beat

def generate_ecg_beat(label=1, noise_level=0.04):
    # P wave
    p = 0.2 * np.exp(-((t - 0.2) ** 2) / (2 * 0.03 ** 2))
    # QRS complex
    q = -0.15 * np.exp(-((t - 0.4) ** 2) / (2 * 0.01 ** 2))
    r = 1.2 * np.exp(-((t - 0.45) ** 2) / (2 * 0.015 ** 2))
    s = -0.3 * np.exp(-((t - 0.5) ** 2) / (2 * 0.015 ** 2))
    # T wave (abnormal if label == -1: inverted or elevated ST)
    if label == 1:
        tw = 0.35 * np.exp(-((t - 0.7) ** 2) / (2 * 0.05 ** 2))
    else: # Ischemia / Infarction ST elevation or inversion
        tw = -0.25 * np.exp(-((t - 0.65) ** 2) / (2 * 0.06 ** 2)) + 0.2 * np.exp(-((t - 0.55)**2)/(2*0.04**2))
    
    baseline = p + q + r + s + tw
    noise = np.random.normal(0, noise_level, len(t))
    return baseline + noise

# Build a continuous multi-beat signal and labeled segments
n_beats = 15
ecg_signals = []
labels = []
timestamps = []
fs = 128.0 # 128 Hz

full_ecg = []
full_labels = []
time_vector = []

current_time = 0.0
for i in range(n_beats):
    lbl = 1 if i % 3 != 0 else -1  # mix normal and ischemia beats
    beat = generate_ecg_beat(label=lbl)
    for sample in beat:
        full_ecg.append(sample)
        full_labels.append("Normal" if lbl == 1 else "Myocardial_Infarction")
        time_vector.append(round(current_time, 4))
        current_time += 1.0 / fs

df_ecg = pd.DataFrame({
    "timestamp": time_vector,
    "ecg_lead_ii": full_ecg,
    "diagnostic_label": full_labels
})
df_ecg.to_csv("data/ecg200_sample.csv", index=False)

# 2. Synthetic Control Time Series
# 6 classes: Normal, Cyclic, Increasing, Decreasing, Upward Shift, Downward Shift
time_axis = np.arange(0, 60)
control_samples = []

for i in range(120): # 20 samples per class
    cls_idx = i % 6
    if cls_idx == 0:
        cls_name = "Normal"
        sig = 30 + np.random.normal(0, 2, 60)
    elif cls_idx == 1:
        cls_name = "Cyclic"
        sig = 30 + 8 * np.sin(2 * np.pi * time_axis / 15) + np.random.normal(0, 1.5, 60)
    elif cls_idx == 2:
        cls_name = "Increasing_Trend"
        sig = 20 + 0.35 * time_axis + np.random.normal(0, 1.5, 60)
    elif cls_idx == 3:
        cls_name = "Decreasing_Trend"
        sig = 40 - 0.35 * time_axis + np.random.normal(0, 1.5, 60)
    elif cls_idx == 4:
        cls_name = "Upward_Shift"
        sig = 25 + np.random.normal(0, 1.5, 60)
        sig[30:] += 12.0
    else:
        cls_name = "Downward_Shift"
        sig = 35 + np.random.normal(0, 1.5, 60)
        sig[30:] -= 12.0
        
    for t_idx, val in enumerate(sig):
        control_samples.append({
            "series_id": f"series_{i:03d}",
            "time_step": t_idx,
            "sensor_value": round(float(val), 4),
            "process_class": cls_name
        })

df_control = pd.DataFrame(control_samples)
df_control.to_csv("data/synthetic_control.csv", index=False)

# 3. Lorenz Attractor Chaotic Signal
def lorenz_system(sigma=10.0, rho=28.0, beta=8.0/3.0, dt=0.01, num_steps=2500):
    xs = np.empty(num_steps)
    ys = np.empty(num_steps)
    zs = np.empty(num_steps)
    xs[0], ys[0], zs[0] = (1.0, 1.0, 1.0)
    for i in range(num_steps - 1):
        dx = sigma * (ys[i] - xs[i]) * dt
        dy = (xs[i] * (rho - zs[i]) - ys[i]) * dt
        dz = (xs[i] * ys[i] - beta * zs[i]) * dt
        xs[i + 1] = xs[i] + dx
        ys[i + 1] = ys[i] + dy
        zs[i + 1] = zs[i] + dz
    return xs, ys, zs

xs, ys, zs = lorenz_system()
t_lorenz = np.arange(len(xs)) * 0.01
df_lorenz = pd.DataFrame({
    "time": np.round(t_lorenz, 3),
    "lorenz_x": np.round(xs, 5),
    "lorenz_y": np.round(ys, 5),
    "lorenz_z": np.round(zs, 5),
    "lorenz_x_noisy_10db": np.round(xs + np.random.normal(0, np.std(xs) * 0.316, len(xs)), 5)
})
df_lorenz.to_csv("data/lorenz_chaos.csv", index=False)

# 4. Mackey-Glass Delayed Differential Equation
def mackey_glass(tau=17, n_samples=2000, dt=0.1, a=0.2, b=0.1, c=10):
    history_len = int(tau / dt)
    x = np.zeros(n_samples + history_len)
    x[:history_len] = 1.2 + 0.1 * np.random.randn(history_len)
    for i in range(history_len, n_samples + history_len - 1):
        x_tau = x[i - history_len]
        dx = (a * x_tau / (1.0 + x_tau ** c) - b * x[i]) * dt
        x[i + 1] = x[i] + dx
    return x[history_len:]

mg_signal = mackey_glass()
t_mg = np.arange(len(mg_signal)) * 0.1
df_mg = pd.DataFrame({
    "time_sec": np.round(t_mg, 2),
    "mackey_glass_signal": np.round(mg_signal, 5)
})
df_mg.to_csv("data/mackey_glass.csv", index=False)

print("Generated benchmark datasets in data/ successfully!")

