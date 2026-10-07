"""Train-only feature reduction used by the released Mitra v2 recipe."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureReducer:
    def __init__(self, problem_type, max_features=256, method="auto", has_categoricals=False):
        if max_features is not None and (not isinstance(max_features, int) or max_features < 1):
            raise ValueError("max_features must be a positive integer or None.")
        if method not in ("auto", "select", "svd", None):
            raise ValueError("feature_reduction must be 'auto', 'select', 'svd', or None.")
        self.problem_type = problem_type
        self.max_features = max_features
        self.method = method
        self.has_categoricals = has_categoricals
        self.columns = None
        self.projection = None

    def fit(self, X: pd.DataFrame, y):
        if (
            self.max_features is None
            or self.method is None
            or X.shape[1] <= self.max_features
            or self.has_categoricals
        ):
            return self
        from sklearn.impute import SimpleImputer

        values = X.to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, np.nan)
        method = self.method
        if method == "auto":
            method = "svd" if self.problem_type == "regression" else "select"
        if method == "svd":
            from sklearn.decomposition import PCA
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            self.projection = make_pipeline(
                SimpleImputer(strategy="mean", keep_empty_features=True),
                StandardScaler(),
                PCA(n_components=min(self.max_features, *values.shape), random_state=0),
            )
            self.projection.fit(values)
        else:
            # Binary descriptor tables retain their full representation.
            continuous = sum(np.unique(column[np.isfinite(column)]).size > 2 for column in values.T)
            if self.problem_type != "regression" and continuous < values.shape[1] * 0.5:
                return self
            from sklearn.feature_selection import f_classif, f_regression

            imputed = SimpleImputer(strategy="mean", keep_empty_features=True).fit_transform(values)
            scorer = f_regression if self.problem_type == "regression" else f_classif
            with np.errstate(divide="ignore", invalid="ignore"):
                scores, _ = scorer(imputed, np.asarray(y))
            scores = np.where(np.isfinite(scores), scores, -np.inf)
            self.columns = X.columns[np.argsort(scores)[::-1][: self.max_features]].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns is not None:
            return X[self.columns]
        if self.projection is not None:
            values = X.to_numpy(dtype=float)
            values = np.where(np.isfinite(values), values, np.nan)
            projected = self.projection.transform(values)
            return pd.DataFrame(projected, index=X.index, columns=[f"svd_{i}" for i in range(projected.shape[1])])
        return X
