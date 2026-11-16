"""
QUANTUM EYE — Signal & Data Preprocessing Module
Includes imputation, outlier handling, scaling, smoothing, filtering, and detrending.
Original data is preserved immutably; all transformation steps are recorded.
"""
import numpy as np
import pandas as pd
from scipy import signal
from typing import Dict, Any, List, Tuple, Optional

class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        # Keep immutable reference to original data
        self._original_df = df.copy()
        self.processed_df = df.copy()
        self.pipeline_history: List[Dict[str, Any]] = []

    @property
    def original_df(self) -> pd.DataFrame:
        return self._original_df.copy()

    def reset(self):
        """Reset processed dataframe back to original state."""
        self.processed_df = self._original_df.copy()
        self.pipeline_history.clear()

    # ----------------- Missing Values -----------------
    def handle_missing(self, column: str, method: str = "interpolate") -> str:
        """
        Handle missing values in a column.
        Methods: 'drop', 'interpolate', 'mean', 'median', 'ffill', 'bfill'
        """
        if column not in self.processed_df.columns:
            return f"Column {column} not found."
            
        initial_missing = int(self.processed_df[column].isnull().sum())
        if initial_missing == 0:
            return f"No missing values in column '{column}'."

        if method == "drop":
            self.processed_df.dropna(subset=[column], inplace=True)
            self.processed_df.reset_index(drop=True, inplace=True)
            desc = f"Dropped {initial_missing} rows with missing values in '{column}'"
        elif method == "interpolate":
            self.processed_df[column] = self.processed_df[column].interpolate(method="linear").bfill().ffill()
            desc = f"Linear interpolation on '{column}' ({initial_missing} imputed)"
        elif method == "mean":
            mean_val = self.processed_df[column].mean()
            self.processed_df[column].fillna(mean_val, inplace=True)
            desc = f"Mean imputation ({mean_val:.4f}) on '{column}'"
        elif method == "median":
            med_val = self.processed_df[column].median()
            self.processed_df[column].fillna(med_val, inplace=True)
            desc = f"Median imputation ({med_val:.4f}) on '{column}'"
        elif method == "ffill":
            self.processed_df[column].ffill().bfill(inplace=True)
            desc = f"Forward fill on '{column}'"
        elif method == "bfill":
            self.processed_df[column].bfill().ffill(inplace=True)
            desc = f"Backward fill on '{column}'"
        else:
            return f"Unknown missing method: {method}"

        self.pipeline_history.append({"step": "missing_value_imputation", "column": column, "method": method, "description": desc})
        return desc

    # ----------------- Duplicates -----------------
    def remove_duplicates(self) -> str:
        """Remove duplicate rows from dataset."""
        n_before = len(self.processed_df)
        self.processed_df.drop_duplicates(inplace=True)
        self.processed_df.reset_index(drop=True, inplace=True)
        n_removed = n_before - len(self.processed_df)
        desc = f"Removed {n_removed} duplicate rows"
        self.pipeline_history.append({"step": "duplicate_removal", "removed": n_removed, "description": desc})
        return desc

    # ----------------- Outliers -----------------
    def detect_outliers(self, column: str, method: str = "iqr", threshold: float = 1.5) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect outliers in column using 'iqr' or 'zscore'.
        Returns boolean mask of outliers and summary metrics.
        """
        vals = self.processed_df[column].dropna().values
        if len(vals) == 0:
            return np.array([]), {}

        if method == "iqr":
            q25, q75 = np.percentile(vals, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - threshold * iqr
            upper_bound = q75 + threshold * iqr
            outlier_mask = (self.processed_df[column] < lower_bound) | (self.processed_df[column] > upper_bound)
            meta = {"method": "IQR", "q25": q25, "q75": q75, "iqr": iqr, "lower": lower_bound, "upper": upper_bound}
        elif method == "zscore":
            mu = np.mean(vals)
            sigma = np.std(vals)
            if sigma < 1e-9:
                return np.zeros(len(self.processed_df), dtype=bool), {"method": "Z-Score", "count": 0}
            z_scores = np.abs((self.processed_df[column] - mu) / sigma)
            outlier_mask = z_scores > threshold
            meta = {"method": "Z-Score", "mean": mu, "std": sigma, "threshold": threshold}
        else:
            raise ValueError(f"Unknown outlier method: {method}")

        count = int(outlier_mask.sum())
        meta["count"] = count
        meta["percentage"] = round((count / len(self.processed_df)) * 100, 2)
        return outlier_mask.values, meta

    def handle_outliers(self, column: str, method: str = "iqr", action: str = "clip", threshold: float = 1.5) -> str:
        """
        Action: 'clip' (winsorize) or 'remove'
        """
        mask, meta = self.detect_outliers(column, method=method, threshold=threshold)
        n_outliers = meta.get("count", 0)
        if n_outliers == 0:
            return f"No outliers detected in '{column}' using {method} (threshold={threshold})."

        if action == "clip":
            if method == "iqr":
                self.processed_df[column] = self.processed_df[column].clip(lower=meta["lower"], upper=meta["upper"])
            else:
                mu, sigma = meta["mean"], meta["std"]
                self.processed_df[column] = self.processed_df[column].clip(lower=mu - threshold * sigma, upper=mu + threshold * sigma)
            desc = f"Clipped {n_outliers} outliers in '{column}' via {method.upper()} bounds."
        elif action == "remove":
            self.processed_df = self.processed_df[~mask].reset_index(drop=True)
            desc = f"Removed {n_outliers} outlier rows in '{column}' via {method.upper()}."
        else:
            return f"Unknown action: {action}"

        self.pipeline_history.append({"step": "outlier_handling", "column": column, "method": method, "action": action, "description": desc})
        return desc

    # ----------------- Scaling & Normalization -----------------
    def scale_signal(self, column: str, method: str = "standard") -> str:
        """
        Methods:
        - 'standard': Z-score standardization (mu=0, std=1)
        - 'minmax': Range normalization to [0, 1]
        - 'robust': Median centering and IQR scaling
        """
        s = self.processed_df[column].values
        if len(s) == 0:
            return "Empty column."

        if method == "standard":
            mu, sigma = np.mean(s), np.std(s)
            if sigma < 1e-9:
                return f"Zero variance in '{column}'; standard scaling skipped."
            self.processed_df[column] = (s - mu) / sigma
            desc = f"Z-score standardized '{column}' (μ={mu:.4f}, σ={sigma:.4f})"
        elif method == "minmax":
            s_min, s_max = np.min(s), np.max(s)
            rng = s_max - s_min
            if rng < 1e-9:
                return f"Zero range in '{column}'; min-max scaling skipped."
            self.processed_df[column] = (s - s_min) / rng
            desc = f"Min-Max normalized '{column}' to [0, 1] (min={s_min:.4f}, max={s_max:.4f})"
        elif method == "robust":
            med = np.median(s)
            q25, q75 = np.percentile(s, [25, 75])
            iqr = q75 - q25
            if iqr < 1e-9:
                iqr = 1.0
            self.processed_df[column] = (s - med) / iqr
            desc = f"Robust scaled '{column}' (median={med:.4f}, IQR={iqr:.4f})"
        else:
            return f"Unknown scaling method: {method}"

        self.pipeline_history.append({"step": "scaling", "column": column, "method": method, "description": desc})
        return desc

    # ----------------- Smoothing & Filtering -----------------
    def apply_moving_average(self, column: str, window: int = 5) -> str:
        """Apply rolling moving-average smoothing."""
        if window < 2:
            return "Window size must be >= 2."
        self.processed_df[column] = self.processed_df[column].rolling(window=window, min_periods=1, center=True).mean()
        desc = f"Moving-average smoothing on '{column}' (window={window})"
        self.pipeline_history.append({"step": "moving_average", "column": column, "window": window, "description": desc})
        return desc

    def apply_median_filter(self, column: str, kernel_size: int = 5) -> str:
        """Apply median filter to suppress impulsive spikes while preserving edges."""
        if kernel_size % 2 == 0:
            kernel_size += 1  # must be odd
        vals = self.processed_df[column].to_numpy(dtype=float)
        filtered = signal.medfilt(vals, kernel_size=kernel_size)
        self.processed_df[column] = filtered
        desc = f"Median filter on '{column}' (kernel={kernel_size})"
        self.pipeline_history.append({"step": "median_filter", "column": column, "kernel": kernel_size, "description": desc})
        return desc

    def apply_savgol_filter(self, column: str, window_length: int = 11, polyorder: int = 3) -> str:
        """Apply Savitzky-Golay polynomial smoothing filter."""
        if window_length % 2 == 0:
            window_length += 1
        if polyorder >= window_length:
            polyorder = window_length - 1
            
        vals = self.processed_df[column].to_numpy(dtype=float)
        if len(vals) < window_length:
            return f"Signal length ({len(vals)}) is smaller than window length ({window_length})."
            
        filtered = signal.savgol_filter(vals, window_length=window_length, polyorder=polyorder)
        self.processed_df[column] = filtered
        desc = f"Savitzky-Golay filter on '{column}' (window={window_length}, poly={polyorder})"
        self.pipeline_history.append({"step": "savgol_filter", "column": column, "window": window_length, "poly": polyorder, "description": desc})
        return desc

    def apply_butterworth_filter(self, column: str, btype: str = "low", cutoff: Any = 10.0, fs: float = 100.0, order: int = 4) -> str:
        """
        Apply Butterworth digital filter.
        btype: 'low', 'high', 'band'
        """
        vals = self.processed_df[column].to_numpy(dtype=float)
        nyq = 0.5 * fs
        if nyq <= 0:
            return "Sampling frequency must be positive."

        try:
            if btype in ["low", "high"]:
                Wn = float(cutoff) / nyq
                if Wn <= 0 or Wn >= 1.0:
                    return f"Cutoff ({cutoff} Hz) must be strictly between 0 and Nyquist ({nyq} Hz)."
                sos = signal.butter(order, Wn, btype=btype, output='sos')
                filtered = signal.sosfiltfilt(sos, vals)
                desc = f"Butterworth {btype}pass filter on '{column}' (cutoff={cutoff}Hz, order={order}, fs={fs}Hz)"
            elif btype == "band":
                if not isinstance(cutoff, (list, tuple)) or len(cutoff) != 2:
                    return "Bandpass requires cutoff tuple (low, high)."
                low_w = float(cutoff[0]) / nyq
                high_w = float(cutoff[1]) / nyq
                if low_w <= 0 or high_w >= 1.0 or low_w >= high_w:
                    return f"Bandpass cutoffs [{cutoff[0]}, {cutoff[1]}] invalid for Nyquist {nyq} Hz."
                sos = signal.butter(order, [low_w, high_w], btype='bandpass', output='sos')
                filtered = signal.sosfiltfilt(sos, vals)
                desc = f"Butterworth bandpass filter on '{column}' ({cutoff[0]}-{cutoff[1]}Hz, order={order})"
            else:
                return f"Unsupported filter type: {btype}"

            self.processed_df[column] = filtered
            self.pipeline_history.append({"step": "butterworth_filter", "column": column, "btype": btype, "cutoff": cutoff, "description": desc})
            return desc
        except Exception as e:
            return f"Butterworth filtering failed: {str(e)}"

    def apply_detrend(self, column: str, detrend_type: str = "linear") -> str:
        """Remove linear trend or constant offset."""
        vals = self.processed_df[column].to_numpy(dtype=float)
        try:
            detrended = signal.detrend(vals, type=detrend_type)
            self.processed_df[column] = detrended
            desc = f"Detrended '{column}' ({detrend_type} detrending)"
            self.pipeline_history.append({"step": "detrend", "column": column, "type": detrend_type, "description": desc})
            return desc
        except Exception as e:
            return f"Detrending failed: {str(e)}"

