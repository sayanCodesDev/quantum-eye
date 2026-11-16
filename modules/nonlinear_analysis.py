"""
QUANTUM EYE — Nonlinear Dynamics & Chaos Engine
Implements Permutation Entropy, Approximate Entropy, Sample Entropy,
Phase Space Reconstruction, Largest Lyapunov Exponent estimation,
Recurrence Plots, and Recurrence Quantification Analysis (RQA).
"""
import math
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Tuple, Optional, List

# ----------------- Shannon Entropy -----------------
def compute_shannon_entropy(signal_values: np.ndarray, num_bins: int = 30) -> float:
    """Computes Shannon entropy across binned amplitude distribution."""
    clean = signal_values[~np.isnan(signal_values)]
    if len(clean) == 0:
        return 0.0
    counts, _ = np.histogram(clean, bins=num_bins)
    p = counts / np.sum(counts)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ----------------- Permutation Entropy (Bandt & Pompe 2002) -----------------
def compute_permutation_entropy(signal_values: np.ndarray, m: int = 3, tau: int = 1) -> Tuple[float, float]:
    """
    Computes Permutation Entropy (PE) and normalized PE.
    m: embedding dimension (typically 3 to 6)
    tau: time delay
    """
    clean = signal_values[~np.isnan(signal_values)]
    n = len(clean)
    if n < m * tau:
        return 0.0, 0.0

    # Extract delay vectors
    n_patterns = n - (m - 1) * tau
    patterns = np.empty((n_patterns, m))
    for i in range(m):
        patterns[:, i] = clean[i * tau : i * tau + n_patterns]

    # Map each vector to ordinal ranking permutation
    permutations = np.argsort(patterns, axis=1)
    
    # Hash unique permutation rows
    _, counts = np.unique(permutations, axis=0, return_counts=True)
    p = counts / np.sum(counts)
    
    pe = float(-np.sum(p * np.log2(p)))
    max_pe = float(np.log2(math.factorial(m)))
    norm_pe = float(pe / max_pe) if max_pe > 0 else 0.0
    return pe, norm_pe


# ----------------- Approximate Entropy (ApEn) & Sample Entropy (SampEn) -----------------
def compute_approximate_entropy(signal_values: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
    """
    Computes Approximate Entropy (ApEn) following Pincus (1991).
    m: embedding dimension
    r: tolerance threshold (default: 0.2 * std)
    """
    clean = signal_values[~np.isnan(signal_values)]
    n = len(clean)
    if n < 30:
        return 0.0

    std = np.std(clean)
    if std < 1e-9:
        return 0.0
    if r is None:
        r = 0.2 * std

    def _phi(dim):
        x = np.array([clean[i : i + dim] for i in range(n - dim + 1)])
        # Max-norm pairwise distance
        # Subsample if n is large for speed (max 1000 points)
        if len(x) > 1000:
            idx = np.linspace(0, len(x) - 1, 1000, dtype=int)
            x_sub = x[idx]
        else:
            x_sub = x
            
        c_i = np.zeros(len(x_sub))
        for i in range(len(x_sub)):
            diff = np.max(np.abs(x - x_sub[i]), axis=1)
            c_i[i] = np.sum(diff <= r) / len(x)
        c_i = c_i[c_i > 0]
        return np.mean(np.log(c_i))

    try:
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        return float(abs(phi_m - phi_m1))
    except Exception:
        return 0.0


def compute_sample_entropy(signal_values: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
    """
    Computes Sample Entropy (SampEn) following Richman & Moorman (2000).
    Eliminates self-matching bias present in ApEn.
    """
    clean = signal_values[~np.isnan(signal_values)]
    n = len(clean)
    if n < 30:
        return 0.0

    std = np.std(clean)
    if std < 1e-9:
        return 0.0
    if r is None:
        r = 0.2 * std

    # Use subsample for quadratic complexity limit
    max_pts = min(n, 1200)
    stride = max(1, n // max_pts)
    s = clean[::stride]
    n_sub = len(s)

    def _count_matches(dim):
        x = np.array([s[i : i + dim] for i in range(n_sub - dim + 1)])
        matches = 0
        total_pairs = 0
        n_vecs = len(x)
        for i in range(n_vecs):
            # Exclude self-match j != i
            diffs = np.max(np.abs(x[i + 1:] - x[i]), axis=1) if i < n_vecs - 1 else np.array([])
            matches += np.sum(diffs <= r)
            total_pairs += len(diffs)
        return matches, total_pairs

    a_matches, _ = _count_matches(m + 1)
    b_matches, _ = _count_matches(m)

    if b_matches == 0 or a_matches == 0:
        return 0.0
    return float(-np.log(a_matches / b_matches))


# ----------------- Phase Space / State Space Reconstruction -----------------
def reconstruct_phase_space(signal_values: np.ndarray, m: int = 3, tau: int = 5) -> np.ndarray:
    """
    Reconstructs state space using Takens' delay embedding theorem.
    Returns array of shape (N - (m-1)*tau, m).
    """
    clean = signal_values[~np.isnan(signal_values)]
    n = len(clean)
    n_vecs = n - (m - 1) * tau
    if n_vecs <= 0:
        return np.empty((0, m))

    embedded = np.empty((n_vecs, m))
    for i in range(m):
        embedded[:, i] = clean[i * tau : i * tau + n_vecs]
    return embedded


def plot_phase_space(embedded: np.ndarray, m: int = 3, title: str = "State Space Reconstruction") -> go.Figure:
    """Generates 2D or 3D interactive phase space attractor trajectory."""
    fig = go.Figure()
    
    # Visual subsample
    n_pts = len(embedded)
    stride = max(1, n_pts // 8000)
    sub = embedded[::stride]

    if m >= 3 and sub.shape[1] >= 3:
        fig.add_trace(go.Scatter3d(
            x=sub[:, 0],
            y=sub[:, 1],
            z=sub[:, 2],
            mode='lines',
            line=dict(
                color=np.arange(len(sub)),
                colorscale='Viridis',
                width=2
            ),
            name='Phase Trajectory'
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=15, color="#0f172a")),
            scene=dict(
                xaxis=dict(title="x(t)", gridcolor="#e2e8f0"),
                yaxis=dict(title="x(t + τ)", gridcolor="#e2e8f0"),
                zaxis=dict(title="x(t + 2τ)", gridcolor="#e2e8f0"),
                bgcolor="#ffffff"
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            height=480
        )
    else:
        fig.add_trace(go.Scatter(
            x=sub[:, 0],
            y=sub[:, 1],
            mode='lines',
            line=dict(color='#0284c7', width=1.5),
            name='2D Trajectory'
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=15, color="#0f172a")),
            xaxis=dict(title="x(t)", gridcolor="#f1f5f9"),
            yaxis=dict(title="x(t + τ)", gridcolor="#f1f5f9"),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            margin=dict(l=40, r=20, t=50, b=40),
            height=420
        )
    return fig


# ----------------- Largest Lyapunov Exponent (Rosenstein / Wolf algorithm) -----------------
def estimate_lyapunov_exponent(signal_values: np.ndarray, m: int = 3, tau: int = 5, max_t: int = 30) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimates Largest Lyapunov Exponent (LLE / lambda_1) using Rosenstein's algorithm:
    Tracks mean logarithmic divergence of nearest neighbors in reconstructed phase space.
    Returns: (lle_estimate, time_steps, mean_divergence)
    """
    embedded = reconstruct_phase_space(signal_values, m=m, tau=tau)
    n_vecs = len(embedded)
    if n_vecs < 100:
        return 0.0, np.array([]), np.array([])

    # Cap size for efficiency
    if n_vecs > 1500:
        embedded = embedded[:1500]
        n_vecs = len(embedded)

    # Find nearest neighbor for each point with Theiler window (temporal separation >= tau)
    theiler_window = tau
    nearest_idx = np.zeros(n_vecs - max_t, dtype=int)
    
    for i in range(n_vecs - max_t):
        dists = np.linalg.norm(embedded - embedded[i], axis=1)
        # Exclude temporal neighbors
        low = max(0, i - theiler_window)
        high = min(n_vecs, i + theiler_window + 1)
        dists[low:high] = np.inf
        nearest_idx[i] = np.argmin(dists)

    # Track divergence over time steps k in [0, max_t]
    divergence = np.zeros(max_t)
    counts = np.zeros(max_t)
    
    for k in range(max_t):
        diffs = []
        for i in range(n_vecs - max_t):
            j = nearest_idx[i]
            if i + k < n_vecs and j + k < n_vecs:
                dist = np.linalg.norm(embedded[i + k] - embedded[j + k])
                if dist > 1e-12:
                    diffs.append(np.log(dist))
        if diffs:
            divergence[k] = np.mean(diffs)
            counts[k] = len(diffs)

    time_steps = np.arange(max_t)
    # Estimate slope over initial linear expansion region
    fit_range = slice(1, min(15, max_t))
    if len(time_steps[fit_range]) > 2 and np.all(counts[fit_range] > 0):
        poly = np.polyfit(time_steps[fit_range], divergence[fit_range], 1)
        lle = float(poly[0])
    else:
        lle = 0.0

    return lle, time_steps, divergence


# ----------------- Recurrence Plot & RQA -----------------
def compute_recurrence_matrix(signal_values: np.ndarray, m: int = 3, tau: int = 3, threshold: Optional[float] = None, max_dim: int = 300) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Computes binary Recurrence Matrix R_{i,j} = Theta(eps - ||x_i - x_j||)
    and Recurrence Quantification Analysis (RQA) statistics.
    """
    clean = signal_values[~np.isnan(signal_values)]
    embedded = reconstruct_phase_space(clean, m=m, tau=tau)
    n = len(embedded)
    if n == 0:
        return np.zeros((1, 1)), {}

    # Downsample phase space if length exceeds max_dim for visual/memory efficiency
    stride = max(1, n // max_dim)
    emb_sub = embedded[::stride]
    n_sub = len(emb_sub)

    # Compute Euclidean distance matrix
    diff = emb_sub[:, np.newaxis, :] - emb_sub[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)

    # Threshold epsilon
    if threshold is None:
        threshold = 0.15 * np.std(dists)

    r_matrix = (dists <= threshold).astype(int)

    # RQA Metrics:
    # Recurrence Rate (RR) = sum(R) / N^2
    rr = float(np.sum(r_matrix) / (n_sub ** 2))

    # Diagonal line analysis for Determinism (DET)
    diag_lengths = []
    for k in range(1, n_sub):
        diag = np.diagonal(r_matrix, offset=k)
        # count contiguous runs of 1s
        cur = 0
        for val in diag:
            if val == 1:
                cur += 1
            else:
                if cur >= 2:
                    diag_lengths.append(cur)
                cur = 0
        if cur >= 2:
            diag_lengths.append(cur)

    if diag_lengths:
        det = float(np.sum(diag_lengths) / max(1, (np.sum(r_matrix) - n_sub)))
        max_diag = int(np.max(diag_lengths))
        # Entropy of diagonal line lengths
        counts, _ = np.histogram(diag_lengths, bins=range(2, max_diag + 2))
        p = counts / np.sum(counts)
        p = p[p > 0]
        rqa_entropy = float(-np.sum(p * np.log2(p)))
    else:
        det = 0.0
        max_diag = 0
        rqa_entropy = 0.0

    rqa_meta = {
        "Recurrence Rate (RR)": rr,
        "Determinism (DET)": min(1.0, det),
        "Max Diagonal Length (L_max)": max_diag,
        "RQA Diagonal Entropy": rqa_entropy,
        "Epsilon Threshold": float(threshold),
        "Matrix Dimension": n_sub
    }

    return r_matrix, rqa_meta


def plot_recurrence_matrix(r_matrix: np.ndarray, title: str = "Recurrence Plot") -> go.Figure:
    """Plots 2D binary recurrence plot heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=r_matrix,
        colorscale=[[0, '#ffffff'], [1, '#0f172a']],
        showscale=False
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#0f172a")),
        xaxis=dict(title="Sample Index i", gridcolor="#f1f5f9"),
        yaxis=dict(title="Sample Index j", gridcolor="#f1f5f9", scaleanchor="x", scaleratio=1),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=50, b=40),
        height=420,
        width=420
    )
    return fig

