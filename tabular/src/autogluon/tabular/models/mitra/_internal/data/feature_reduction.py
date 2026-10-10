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
        self.input_columns = None
        self._training_projection = None
        self._training_keys = None

    def fit(self, X: pd.DataFrame, y, *, cache_training=False):
        if (
            self.max_features is None
            or self.method is None
            or X.shape[1] <= self.max_features
            or self.has_categoricals
        ):
            return self
        from sklearn.impute import SimpleImputer

        values = X.to_numpy(dtype=float)
        method = self.method
        if method == "auto":
            method = "svd" if self.problem_type == "regression" else "select"
        if method == "svd":
            from sklearn.decomposition import PCA
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            self.projection = make_pipeline(
                SimpleImputer(strategy="mean"),
                StandardScaler(),
                PCA(n_components=min(self.max_features, *values.shape), random_state=0),
            )
            # PCA.fit_transform and PCA.transform need not use the same
            # numerical computation, especially for randomized SVD.
            projected = self.projection.fit_transform(values)
            self.input_columns = X.columns.tolist()
            if cache_training:
                keys = self._row_keys(X)
                unique = ~keys.duplicated()
                self._training_keys = keys[unique]
                self._training_projection = projected[unique]
        else:
            # Binary descriptor tables retain their full representation.
            values = np.where(np.isfinite(values), values, np.nan)
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

    @staticmethod
    def _row_keys(X):
        # Include values and index: separate tables may reuse a RangeIndex.
        return pd.Index(pd.util.hash_pandas_object(X.astype(float), index=True).to_numpy())

    def clear_training_cache(self):
        self._training_projection = None
        self._training_keys = None

    @staticmethod
    def _frame(projected, X):
        return pd.DataFrame(projected, index=X.index, columns=[f"svd_{i}" for i in range(projected.shape[1])])

    def transform_training(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reuse fit_transform rows during fitting and fold validation only."""
        if self._training_projection is None:
            return self.transform(X)
        positions = self._training_keys.get_indexer(self._row_keys(X[self.input_columns]))
        if (positions < 0).any():
            # AutoGluon can append pseudolabeled rows and reset the fold's
            # index. Those rows have no cached fit_transform result; apply
            # the already fitted projection without refitting it.
            return self.transform(X)
        return self._frame(self._training_projection[positions], X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns is not None:
            return X[self.columns]
        if self.projection is not None:
            values = X[self.input_columns].to_numpy(dtype=float)
            projected = self.projection.transform(values)
            return self._frame(projected, X)
        return X
