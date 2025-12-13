from typing import Literal

import numpy as np
import pandas as pd


def dataset_magnitude(X_in: pd.DataFrame, method: Literal["rms", "max", "median"] = "rms") -> float:
    """
    Compute a single magnitude statistic over all numeric values in the dataset,
    ignoring NaNs.
    """
    X = X_in.copy()
    Xv = X.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if Xv.size == 0:
        return 1.0
    if method == "rms":
        return float(np.sqrt(np.nanmean(Xv**2)))
    elif method == "max":
        return float(np.nanmax(np.abs(Xv)))
    elif method == "median":
        return float(np.nanmedian(np.abs(Xv)))
    else:
        raise ValueError("method must be 'rms', 'max', or 'median'")


def global_scale_preserve_ops(
    X_in: pd.DataFrame,
    target: Literal["rms", "max", "median"] = "rms",
    target_value: float = 1.0,
    out_dtype: np.dtype = np.float64,
) -> tuple[pd.DataFrame, float, float]:
    """
    Scale the entire dataset by a single positive constant a, preserving
    +, -, *, / correlations exactly (NaNs preserved). Then cast to out_dtype.
    """
    X = X_in.copy()
    M = dataset_magnitude(X, method=target)  # NaN-aware
    a = 1.0 if not np.isfinite(M) or M == 0.0 else (target_value / M)
    # Apply to all numeric columns only; NaNs remain NaN
    Xs = X.copy()
    num_cols = Xs.select_dtypes(include=[np.number]).columns
    Xs[num_cols] = (Xs[num_cols].to_numpy(dtype=np.float64) * a).astype(out_dtype)
    return Xs, a, M


def minimize_numeric_dtypes(X):
    """Safely downcast all columns in X to minimal possible dtypes without overflow."""
    X_opt = X.copy()

    int_types = [np.int8, np.int16, np.int32, np.int64]
    float_types = [np.float16, np.float32, np.float64]

    for col in X_opt.columns:
        s = X_opt[col]
        if pd.api.types.is_integer_dtype(s):
            cmin, cmax = s.min(), s.max()
            for t in int_types:
                if np.iinfo(t).min <= cmin and cmax <= np.iinfo(t).max:
                    X_opt[col] = s.astype(t)
                    break

        elif pd.api.types.is_float_dtype(s):
            cmin, cmax = s.min(skipna=True), s.max(skipna=True)
            for t in float_types:
                finfo = np.finfo(t)
                if np.isfinite(cmin) and np.isfinite(cmax):
                    if finfo.min <= cmin and cmax <= finfo.max:
                        X_opt[col] = s.astype(t)
                        break
                else:
                    X_opt[col] = s.astype(np.float32)
                    break

    return X_opt


def reduce_memory_usage(X_in: pd.DataFrame, verbose: bool = True, rescale: bool = True) -> pd.DataFrame:
    """Reduce memory usage of DataFrame by downcasting numeric types.
    Parameters
    ----------
    X_in : pd.DataFrame
        Input DataFrame.
    verbose : bool, default=True
        Whether to print memory usage before and after optimization.
    rescale : bool, default=True
        Whether to rescale numeric values before downcasting to preserve numeric stability.
    Returns
    -------
    pd.DataFrame
        Optimized DataFrame with reduced memory usage.
    """
    X_out = X_in.copy()
    if verbose:
        print(f"Memory before: {X_in.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    if rescale:
        X_out, a, M = global_scale_preserve_ops(X_out, target="rms", target_value=1000, out_dtype=np.float64)
    X_out = minimize_numeric_dtypes(X_out)
    if verbose:
        print(f"Memory after:  {X_out.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    return X_out
