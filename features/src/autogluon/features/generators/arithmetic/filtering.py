import numpy as np
import pandas as pd
from numba import njit, prange
import re
from typing import Literal, Optional
# import numexpr as ne
'''
Further filtering ideas:
- based on target_corr - if its the same, the feature is likely to contain the same info
'''

def remove_mostlynan_features(X: pd.DataFrame) -> pd.DataFrame:
    return X.loc[:, X.isna().mean() < 0.99]

def remove_constant_features(X: pd.DataFrame) -> pd.DataFrame:
    return X.loc[:, X.astype("float64").std() > 0] # float64 to avoid overflow warning

def remove_same_range_features(X: pd.DataFrame, x: pd.Series) -> float:
    # TODO: Change function name to be more descriptive of what actually happens
    col = x.name
    feature_names = [f for f in re.split(r'_(\*|/|\+|\-)_', col) if f not in {'*', '/', '+', '-'}]

    return X[feature_names].corrwith(x, method='spearman').max()

def basic_filter(
        X_in: pd.DataFrame, 
        # y_in: pd.Series,
        min_cardinality: int = 3,
        candidate_cols: list = None,
        use_polars: bool = False,
        remove_constant_mostlynan: bool = True,
        ) -> list:
    """
    Basic filtering of base and generated features:
    - Remove features with cardinality < min_cardinality
    - Optionally remove features that are constant or mostly NaN
    Parameters
    ----------
    X_in : pd.DataFrame
        Input features to filter
    min_cardinality : int, default=3
        Minimum cardinality required to keep a feature
    candidate_cols : list, default=None
        If provided, only these columns will be considered for filtering
    use_polars : bool, default=False
        Whether to use Polars for filtering (faster for large datasets)
    remove_constant_mostlynan : bool, default=True
        Whether to remove features that are nearly constant or mostly NaN
    Returns
    -------
    pd.DataFrame
        Filtered features   
    """
    X = X_in.copy()
    
    # Filter by minimum cardinality
    X = X.loc[:, X.nunique() >= min_cardinality]

    if X.empty:
        return X

    # if predetermined candidate columns are given, use them
    if candidate_cols is not None:
        X = X[candidate_cols]

    if remove_constant_mostlynan:
        if use_polars:
            from .filtering_polars import remove_mostlynan_features_pl, remove_constant_features_pl
            X = remove_mostlynan_features_pl(X)
            X = remove_constant_features_pl(X)
        else:
            X = remove_mostlynan_features(X)
            # TODO: Think whether we need this, was uncommented previously
            X = remove_constant_features(X)

    return X

def fast_spearman(X: pd.DataFrame) -> pd.DataFrame:
    """ Compute Spearman correlation matrix using rank transformation and pairwise Pearson correlation. """
    # 1) Rank in pandas to match tie handling exactly
    R = X.rank(method="average", na_option="keep")
    A = R.to_numpy(float)
    p = A.shape[1]
    # A = R.to_numpy(dtype=np.float32) # Could be float32 for less memory, but float64 is more accurate
    C = _pearson_pairwise_nan(A)           # numba-accelerated pairwise corr
    return pd.DataFrame(C, index=X.columns, columns=X.columns)

@njit(parallel=True, fastmath=False)
def _pearson_pairwise_nan(A: np.ndarray) -> np.ndarray:
    """ Compute pairwise Pearson correlation with NaN handling. """
    n, p = A.shape
    out = np.empty((p, p), dtype=np.float64)

    # diagonals first
    for i in prange(p):
        out[i, i] = 1.0

    # upper triangle
    for i in prange(p):
        xi = A[:, i]
        for j in range(i + 1, p):
            x = xi
            y = A[:, j]

            # pairwise mask (ignore NaNs)
            m = (~np.isnan(x)) & (~np.isnan(y))
            cnt = np.sum(m)
            if cnt < 2:
                out[i, j] = np.nan
                continue

            xm = np.mean(x[m])
            ym = np.mean(y[m])
            dx = x[m] - xm
            dy = y[m] - ym
            num = np.sum(dx * dy)
            den = np.sqrt(np.sum(dx * dx) * np.sum(dy * dy))
            out[i, j] = num / den if den > 0 else np.nan

    # mirror to lower triangle
    for i in prange(p):
        for j in range(i):
            out[i, j] = out[j, i]

    return out

def drop_high_corr(corr: pd.DataFrame, corr_threshold: float = 0.9) -> list:
    """ Identify columns to drop based on a correlation threshold. """
    ac = corr.abs().copy()
    np.fill_diagonal(ac.values, 0.0)
    upper = ac.where(np.triu(np.ones(ac.shape), k=1).astype(bool))
    return upper.gt(corr_threshold).any(axis=0)[lambda s: s].index.tolist()


def filter_by_spearman(X: pd.DataFrame, corr_threshold: float=0.95) -> list:
    """ Filtering of a base feature set based on correlation. """
    spearman_corr = fast_spearman(X)
    np.fill_diagonal(spearman_corr.values, 0)
    drop_cols = drop_high_corr(spearman_corr, corr_threshold=corr_threshold)
    return [col for col in X.columns if col not in drop_cols]

def cross_spearman(df1, df2):
    """
    Compute Spearman correlations between all features in df1 vs df2.
    Missing values allowed. Pairwise-complete observations used.
    """
    # ---- 1) Rank-transform dataframes ----
    r1 = df1.rank(axis=0, na_option='keep')
    r2 = df2.rank(axis=0, na_option='keep')

    r1 = r1.to_numpy(float)
    r2 = r2.to_numpy(float)

    n1 = r1.shape[1]
    n2 = r2.shape[1]

    # ---- 2) Masks for non-missing ranks ----
    mask1 = ~np.isnan(r1)
    mask2 = ~np.isnan(r2)

    # ---- 3) Center ranks but keep NaN in place ----
    # Need nanmean to avoid biasing mean when NaNs present
    r1_centered = r1 - np.nanmean(r1, axis=0)
    r2_centered = r2 - np.nanmean(r2, axis=0)

    # Replace NaNs with zero so dot products ignore them
    r1c = np.where(mask1, r1_centered, 0.0)
    r2c = np.where(mask2, r2_centered, 0.0)

    # ---- 4) Compute pairwise sample sizes (valid rows per pair) ----
    # Broadcast masks: (samples × n1 × n2)
    n_valid = mask1[:,:,None] & mask2[:,None,:]
    n_valid = n_valid.sum(axis=0)   # shape (n1, n2)

    # ---- 5) Compute covariance using dot product but divide by valid count-1 ----
    cov = (r1c.T @ r2c) / (n_valid - 1)

    # ---- 6) Compute std dev for each feature using valid data ----
    # std = sqrt( sum((x-mean)^2) / (n_valid-1) )
    ss1 = (r1c**2).sum(axis=0)      # sum of squares for each column
    ss2 = (r2c**2).sum(axis=0)

    std1 = np.sqrt(ss1 / (mask1.sum(axis=0) - 1))  # shape (n1,)
    std2 = np.sqrt(ss2 / (mask2.sum(axis=0) - 1))

    # Outer-normalize to get Spearman correlations
    corr = cov / np.outer(std1, std2)

    # Where n_valid < 2 → correlation is undefined
    corr[n_valid < 2] = np.nan

    return pd.DataFrame(corr, index=df1.columns, columns=df2.columns)

def filter_by_cross_correlation(
        X_base: pd.DataFrame, 
        X_new: pd.DataFrame, 
        corr_threshold: float=0.95
    ) -> list:
    """ Filter new features by cross-correlation with base features. """
    cross_corr = cross_spearman(X_base, X_new)
    to_drop = set()
    for new_col in X_new.columns:
        if (cross_corr[new_col].abs() > corr_threshold).any():
            to_drop.add(new_col)
    to_keep = [col for col in X_new.columns if col not in to_drop]
    novelty_scores = 1-cross_corr[to_keep].abs().max() # Higher --> more novel
    return to_keep, novelty_scores
        