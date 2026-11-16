"""
QUANTUM EYE — Data Loader & Inspector Module
Supports CSV, Excel (.xlsx, .xls), JSON with automated inspection,
type inference, timestamp detection, and validation.
"""
import io
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List

def load_dataset(file_obj_or_path, file_name: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load dataset from an uploaded file object or file path.
    Returns (DataFrame, error_message).
    """
    try:
        if isinstance(file_obj_or_path, str):
            name = file_name or os.path.basename(file_obj_or_path)
            ext = os.path.splitext(name)[1].lower()
            if ext == '.csv':
                df = pd.read_csv(file_obj_or_path)
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_obj_or_path)
            elif ext == '.json':
                df = pd.read_json(file_obj_or_path)
            else:
                # Attempt CSV fallback
                df = pd.read_csv(file_obj_or_path)
        else:
            name = file_name or getattr(file_obj_or_path, 'name', 'uploaded_file.csv')
            ext = os.path.splitext(name)[1].lower()
            if ext == '.csv':
                df = pd.read_csv(file_obj_or_path)
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_obj_or_path)
            elif ext == '.json':
                df = pd.read_json(file_obj_or_path)
            else:
                try:
                    df = pd.read_csv(file_obj_or_path)
                except Exception:
                    file_obj_or_path.seek(0)
                    df = pd.read_excel(file_obj_or_path)

        if df.empty:
            return None, "Uploaded dataset contains no data (empty table)."
        
        return df, None
    except Exception as e:
        return None, f"Failed to load dataset: {str(e)}"


def inspect_dataset(df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict[str, Any]:
    """
    Performs comprehensive automated scientific inspection of the dataset.
    """
    n_rows, n_cols = df.shape
    columns = list(df.columns)
    
    # Identify column data types
    dtypes = {col: str(df[col].dtype) for col in columns}
    
    # Numerical vs categorical
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in columns if c not in num_cols]
    
    # Missing values
    missing_counts = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    missing_pct = round((total_missing / (n_rows * n_cols)) * 100, 2) if n_rows * n_cols > 0 else 0.0
    
    # Duplicate rows
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / n_rows) * 100, 2) if n_rows > 0 else 0.0
    
    # Identify potential timestamp columns
    timestamp_candidates = detect_timestamp_columns(df)
    
    # Summary stats for numerical columns
    summary_stats = df[num_cols].describe().T if num_cols else pd.DataFrame()
    
    # Check if dataset appears to be sequential time series
    is_sequential, seq_reason = assess_sequentiality(df, timestamp_candidates, num_cols)
    
    return {
        "dataset_name": dataset_name,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": columns,
        "dtypes": dtypes,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "missing_counts": missing_counts,
        "total_missing": total_missing,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": duplicate_pct,
        "timestamp_candidates": timestamp_candidates,
        "summary_stats": summary_stats,
        "is_sequential": is_sequential,
        "seq_reason": seq_reason
    }


def detect_timestamp_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect columns that likely represent time, date, or monotonic sampling indices.
    """
    candidates = []
    time_keywords = ['time', 'date', 'timestamp', 'datetime', 'epoch', 'sec', 'second', 'ms', 'hour', 'day']
    
    for col in df.columns:
        col_lower = str(col).lower()
        # Keyword match
        if any(kw in col_lower for kw in time_keywords):
            candidates.append(col)
            continue
            
        # Datetime dtype
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            candidates.append(col)
            continue
            
        # Try parsing small string sample as datetime
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            sample = df[col].dropna().head(10)
            if len(sample) > 0:
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(sample, errors='raise')
                    candidates.append(col)
                    continue
                except (ValueError, TypeError):
                    pass
                    
        # Monotonically strictly increasing numeric column
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna()
            if len(s) > 10 and s.is_monotonic_increasing:
                candidates.append(col)
                
    return candidates


def assess_sequentiality(df: pd.DataFrame, timestamp_candidates: List[str], num_cols: List[str]) -> Tuple[bool, str]:
    """
    Scientifically assesses whether the dataset represents a sequential signal or purely static tabular records.
    """
    if timestamp_candidates:
        return True, f"Found candidate temporal index: '{timestamp_candidates[0]}'."
    
    if len(num_cols) == 0:
        return False, "No numerical columns detected in dataset."
        
    # Test first numerical column for serial autocorrelation
    sample_col = num_cols[0]
    series = df[sample_col].dropna().values
    if len(series) < 30:
        return False, "Dataset contains fewer than 30 samples; insufficient for meaningful signal dynamics."
        
    # Check lag-1 autocorrelation
    s_std = np.std(series)
    if s_std < 1e-9:
        return False, "Signal is constant (zero variance)."
        
    norm_s = series - np.mean(series)
    autocorr_1 = np.corrcoef(norm_s[:-1], norm_s[1:])[0, 1]
    
    if np.isnan(autocorr_1):
        return False, "Statistical properties cannot be computed."
        
    if abs(autocorr_1) > 0.25:
        return True, f"High lag-1 serial autocorrelation ({autocorr_1:.3f}) suggests underlying temporal continuity."
    else:
        return False, f"Low lag-1 autocorrelation ({autocorr_1:.3f}). Samples may represent independent static observations rather than continuous time series."
