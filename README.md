# QUANTUM EYE — BEYOND RANDOMNESS
### Machine Learning for Intelligent Complex Signal Analysis

![Quantum Eye Architecture](https://img.shields.io/badge/Status-Operational%20Prototype-0284c7)
![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue)
![Streamlit](https://img.shields.io/badge/Interface-Streamlit-ff4b4b)
![PennyLane](https://img.shields.io/badge/Quantum-PennyLane%200.45-8b5cf6)
![License](https://img.shields.io/badge/License-MIT-teal)

---

## 🔬 Project Overview

**QUANTUM EYE** is an interactive scientific laboratory where researchers, data scientists, and engineers can upload almost any numerical time series or static tabular dataset and automatically discover hidden mathematical patterns, distinguish complex signal regimes (periodic, stochastic, chaotic), isolate transient anomalies, and explore experimental quantum feature mapping.

This platform is a **genuine working scientific software prototype**, grounded in rigorous mathematical implementations of digital signal processing, nonlinear dynamics (Takens' delay embedding, Lyapunov exponents, recurrence quantification), classical machine learning, and quantum kernel estimation using PennyLane.

---

## 🚀 Key Laboratory Capabilities

1. **Dataset Ingestion & Automated Inspection**
   - Ingest CSV, Excel (`.xlsx`, `.xls`), and JSON files.
   - Automated profiling: row/column dimensions, datatypes, missing value counts, duplicate rows, descriptive moments.
   - Automatic timestamp detection and sequentiality assessment (serial autocorrelation testing).
   - Pre-loaded benchmark datasets from the **UCR/UEA Time Series Classification Archive** ([timeseriesclassification.com](https://www.timeseriesclassification.com/)).

2. **Interactive Scientific Preprocessing**
   - Missing-value handling: linear interpolation, mean/median imputation, forward/backward fill.
   - Outlier detection via IQR bounds and Z-Score limits (winsorizing/clipping or row purging).
   - Amplitude scaling: Z-Score standardization ($\mu=0, \sigma=1$), Min-Max scaling ($[0, 1]$), and Robust scaling (Median/IQR).
   - Filtering: Moving average, Median filter, Savitzky-Golay polynomial smoothing, digital Butterworth filter (low-pass, high-pass, band-pass), and linear/constant detrending.
   - **Immutable original data**: All transformations are recorded in a pipeline log with real-time before-vs-after overlay graphs.

3. **Time-Domain Signal Analysis**
   - Statistical moments: Mean, Median, Variance, Standard Deviation, Skewness, Excess Kurtosis, RMS amplitude, Peak-to-Peak amplitude, Crest factor, Zero-crossing rate.
   - Dynamic rolling mean with $\pm 2\sigma$ uncertainty envelopes.
   - Gaussian probability density fit and Box plot spread.

4. **Frequency-Domain Spectral Analysis**
   - Fast Fourier Transform (FFT) one-sided amplitude spectrum.
   - Welch's Power Spectral Density (PSD) periodogram in linear and logarithmic (dB/Hz) scales.
   - Dominant peak tracking, Spectral Centroid, Spectral Spread, and Spectral Entropy.
   - *Physical Unit Guardrail*: If sampling frequency ($F_s$) is unknown, normalized frequency (cycles/sample) is strictly used to prevent false Hertz (Hz) claims.

5. **Nonlinear Dynamics & Chaos Laboratory**
   - Permutation Entropy (Bandt & Pompe, 2002) with configurable embedding dimension $m$ and delay $\tau$.
   - Approximate Entropy (ApEn) and Sample Entropy (SampEn).
   - Largest Lyapunov Exponent ($\lambda_1$) estimation via Rosenstein's phase-space trajectory divergence algorithm.
   - 2D and 3D State-Space Attractor Reconstruction via Takens' delay embedding theorem.
   - Recurrence Plot (RP) matrix and Recurrence Quantification Analysis (RQA: Recurrence Rate, Determinism, Diagonal Entropy).

6. **Built-in Signal Generator & Mixer**
   - Periodic waveforms: Sine, Square, Sawtooth, and Composite multi-harmonics.
   - Stochastic noise: Gaussian white noise, Uniform noise, Pink noise ($1/f$), and Brown noise ($1/f^2$).
   - Deterministic chaotic systems: Lorenz Attractor ($\dot{x}, \dot{y}, \dot{z}$), Logistic Map ($r$), and Rössler Attractor.
   - Hybrid Signal Mixer: Blend multiple waveforms and inject Gaussian noise with calibrated Signal-to-Noise Ratio (SNR in dB).
   - One-click CSV export and instant ingestion into the laboratory workspace.

7. **Classical Machine Learning Engine**
   - **Supervised Classifiers**: Random Forest, Support Vector Machine (Linear / RBF), Gradient Boosting, Decision Tree, K-Nearest Neighbors.
   - Strict leakage-free scaling: Scalers and imputation fit exclusively on the training split.
   - Stratified train/test splitting, cross-validation, confusion matrices, and precision/recall/F1 metrics.
   - **Unsupervised Learning**: K-Means clustering, DBSCAN, and 2D/3D Principal Component Analysis (PCA).
   - **Controlled 3-Class Signal Benchmark**: Classifies independently generated Periodic, Stochastic, and Chaotic waveforms with zero overlapping window leakage.

8. **Anomaly Detection Intelligence**
   - Algorithms: Isolation Forest, Z-Score thresholding, Rolling Deviation envelope, Robust Median Absolute Deviation (MAD), and One-Class SVM.
   - High-contrast interactive waveform visualization highlighting anomalous timestamps and samples.
   - Continuous anomaly score trajectory and probability distribution.
   - Sensitivity thresholds and transparent scientific caveats.

9. **Quantum Machine Learning Laboratory**
   - Experimental Quantum Computing component powered by **PennyLane 0.45**.
   - Quantum Kernel Support Vector Machine (QSVM) utilizing the **ZZ Feature Map** with Hadamard gates, $R_z$ rotations, and CNOT entanglement.
   - Computes quantum fidelity Gram matrix: $K_{ij} = |\langle\phi(x_i)|\phi(x_j)\rangle|^2$.
   - Telemetry: Reports exact simulator backend (`default.qubit` / `lightning.qubit`), qubit count ($2-4$), circuit depth, gate evaluations, and quantum computational overhead.
   - *Zero Fabrication Guarantee*: Transparent evaluation without artificial speedup claims.

10. **Classical vs. Quantum Comparison**
    - Side-by-side benchmarking of Accuracy, Precision, Recall, F1, training latency, and inference time.
    - **Noise Robustness Experiment**: Evaluates accuracy degradation across SNR levels (Clean $\infty$ dB, 25 dB, 20 dB, 15 dB, 10 dB, 5 dB).

11. **Model Explainability & Reporting**
    - **SHAP (SHapley Additive exPlanations)** feature attribution bar chart and ranking table.
    - **Scientific Synthesis Engine**: Structured multi-domain findings interpreted via a deterministic rule-based scientific reasoning engine (with optional Google Gemini or OpenAI API integration).
    - **Downloadable PDF Report**: Compiles dataset metadata, preprocessing history, moments, spectral properties, nonlinear dynamics, anomaly counts, and model comparisons into a publication-ready PDF using ReportLab.

---

## 📁 Repository Structure

```
ecomcp/
├── app.py                     # Main Streamlit application entrypoint & router
├── config.py                  # Theme colors, styling constants, and limits
├── requirements.txt           # Verified Python package dependencies
├── README.md                  # Comprehensive documentation and guide
├── generate_datasets.py       # Benchmark dataset generator
├── assets/
│   └── style.css              # Custom scientific styling (white/navy/cyan theme)
├── data/                      # Benchmark time-series datasets
│   ├── ecg200_sample.csv      # ECG Heartbeat (Normal vs. Myocardial Infarction)
│   ├── synthetic_control.csv  # 6-class industrial control chart patterns
│   ├── lorenz_chaos.csv       # Deterministic Lorenz chaotic attractor (x, y, z)
│   └── mackey_glass.csv       # Delayed differential equation series
├── modules/
│   ├── data_loader.py         # CSV, Excel, JSON loader & automated inspector
│   ├── preprocessing.py       # Filtering, scaling, imputation, and detrending
│   ├── signal_generator.py    # Periodic, stochastic, and chaotic generators
│   ├── signal_analysis.py     # Time-domain moments and Plotly visualizers
│   ├── frequency_analysis.py  # FFT, Welch PSD, centroid, and spectral entropy
│   ├── nonlinear_analysis.py  # Entropies, Lyapunov exponent, phase space, RQA
│   ├── feature_extraction.py  # Unified statistical, frequency, and nonlinear extractor
│   ├── ml_engine.py           # Supervised classifiers, unsupervised PCA, 3-class test
│   ├── anomaly_detection.py   # Isolation Forest, Z-score, MAD, One-Class SVM
│   ├── quantum_engine.py      # PennyLane Quantum Kernel SVM & circuit telemetry
│   ├── explainability.py      # SHAP feature importance calculator
│   ├── llm_interpreter.py     # Deterministic scientific synthesis & LLM API hook
│   ├── report_generator.py    # Multi-page ReportLab PDF report compiler
│   └── state_manager.py       # Centralized session state coordinator
└── pages/
    ├── p01_overview.py            # Overview KPI dashboard
    ├── p02_upload_dataset.py      # File upload & inspection
    ├── p_generator.py             # Signal generator & hybrid mixer
    ├── p03_dataset_explorer.py    # Data table & preprocessing
    ├── p04_signal_analysis.py     # Time-domain waveform & rolling bands
    ├── p05_statistical_analysis.py# Autocorrelation, cross-corr, heatmap
    ├── p06_frequency_analysis.py  # FFT & Welch Power Spectral Density
    ├── p07_nonlinear_dynamics.py  # Lyapunov, phase space, recurrence plots
    ├── p08_machine_learning.py    # Supervised, unsupervised & 3-class experiment
    ├── p09_anomaly_detection.py   # Anomaly highlighting & score distributions
    ├── p10_quantum_lab.py         # Quantum Kernel SVM & circuit diagnostics
    ├── p11_model_comparison.py    # Classical vs Quantum & SNR sweep
    └── p12_reports.py             # SHAP, LLM interpretation & PDF export
```

---

## ⚙️ Installation & Quick Start

### 1. Prerequisites
- Python 3.11 or Python 3.12 (recommended for full pre-compiled wheel compatibility).
- A standard laptop CPU (no GPU required).

### 2. Virtual Environment Setup

Using `uv` (fastest):
```bash
# Clone or navigate to the repository directory
cd /path/to/ecomcp

# Create a Python 3.12 virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

Or using standard `python3` venv:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Generate Benchmark Datasets
If the `data/` folder is empty, generate the benchmark series:
```bash
python generate_datasets.py
```

### 4. Run the Web Application
```bash
streamlit run app.py
```
Open your web browser at `http://localhost:8501`.

---

## 📊 Scientific Datasets & External References

Quantum Eye incorporates sample datasets inspired by the gold-standard time-series machine learning repository:

- **UCR/UEA Time Series Classification Archive**: [https://www.timeseriesclassification.com/](https://www.timeseriesclassification.com/)
  - *ECG200*: Single-lead electrocardiogram tracing discriminating normal heartbeats from myocardial infarction ischemia.
  - *Synthetic Control*: Standardized process chart benchmark featuring Normal, Cyclic, Increasing trend, Decreasing trend, Upward shift, and Downward shift dynamics.
  - *Lorenz Attractor (1963)*: Classic 3-variable atmospheric convection model ($\dot{x} = \sigma(y-x), \dot{y} = x(\rho-z)-y, \dot{z} = xy-\beta z$) demonstrating sensitive dependence on initial conditions.
  - *Mackey-Glass (1977)*: Nonlinear delayed differential equation ($\frac{dx}{dt} = \frac{a x_\tau}{1 + x_\tau^{10}} - b x$) modeling physiological feedback control.

---

## 🛡️ Scientific Transparency & Limitations

- **No False Claims**: Mathematical measures (e.g., positive Lyapunov exponent, high permutation entropy) provide empirical evidence of complexity, but do not singularly prove chaos in the presence of colored noise or non-stationarity.
- **Physical Frequency Guardrails**: When sampling frequency ($F_s$) is unknown, frequency scales are presented as normalized cycles per sample; Hz units are never falsely attributed.
- **Simulated Quantum Mechanics**: Quantum kernel evaluations execute on local state-vector simulators (`default.qubit`). Classical simulation overhead is openly reported alongside accuracy metrics.
- **Data Leakage Prevention**: All preprocessing transformations and standardization parameters are fitted exclusively on training partitions.

