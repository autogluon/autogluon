from typing import Literal, Tuple

import numpy as np
import pandas as pd

from autogluon.common.features.types import (
    R_BOOL,
    R_CATEGORY,
    R_FLOAT,
    R_INT,
    R_OBJECT,
    S_DATETIME_AS_OBJECT,
    S_IMAGE_BYTEARRAY,
    S_IMAGE_PATH,
    S_TEXT,
    S_TEXT_AS_CATEGORY,
)

from .abstract import AbstractFeatureGenerator


class FrequencyFeatureGenerator(AbstractFeatureGenerator):
    """
    Generate frequency encoded features for categorical variables.
    Parameters
    ----------
    target_type : str or None, default=None
        The type of the target variable ('regression', 'binary', 'multiclass')
    keep_original : bool, default=True
        Whether to keep the original features.
    only_categorical : bool, default=True
        Whether to only apply frequency encoding to categorical features.
    candidate_cols : list of str or None, default=None
        List of candidate columns to consider for frequency encoding. If None, all columns are used.
    use_filters : bool, default=True
        Whether to filter candidate columns based on distinctiveness.
    fillna : int or None, default=0
        Value to fill NaNs in frequency encoding. If None, NaNs are kept as NaN.
    log : bool, default=False
        Whether to apply log transformation to frequency encoded values.
    **kwargs
        Additional keyword arguments.
    Returns
    -------
    self : FrequencyFeatureGenerator
        Fitted FrequencyFeatureGenerator instance.
    """

    def __init__(
        self,
        target_type: Literal["regression", "binary", "multiclass"] | None = None,
        keep_original: bool = True,
        only_categorical: bool = True,
        candidate_cols: list = None,
        use_filters: bool = True,
        fillna: int | None = 0,
        log: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_type = target_type
        self.keep_original = keep_original  # TODO: Clarify if and how something similar to keep_original logic is already in AG preprocessors
        self.only_categorical = only_categorical
        self.candidate_cols = candidate_cols
        self.use_filters = use_filters
        self.log = log

        if fillna is None:
            self.fillna = np.nan
        else:
            self.fillna = fillna

        self.freq_maps = {}
        self.passthrough_cols = []

    def estimate_no_of_new_features(self, X: pd.DataFrame, **kwargs) -> int:
        # TODO: Improve estimation using other hyperparameters
        # TODO: Account for the fact that some columns may be removed if keep_original is False
        if self.only_categorical:
            return X.select_dtypes(include=["object", "category"]).shape[1]
        else:
            return X.shape[1]

    @classmethod
    def filter_candidates_by_distinctiveness(cls, X: pd.DataFrame) -> list:
        candidate_cols = []
        for col in X.columns:
            x_new = X[col].map(X[col].value_counts().to_dict())
            if all((pd.crosstab(X[col], x_new) > 0).sum() == 1):
                continue
            else:
                candidate_cols.append(col)

        return candidate_cols

    def _fit(self, X_in: pd.DataFrame, y_in: pd.Series = None):
        X = X_in.copy()

        if self.candidate_cols is not None:
            self.passthrough_cols.extend([col for col in X.columns if col not in self.candidate_cols])
            X = X[self.candidate_cols]
        if self.only_categorical:
            self.passthrough_cols.extend(
                [col for col in X.columns if col not in X.select_dtypes(include=["object", "category"]).columns]
            )
            X = X.select_dtypes(include=["object", "category"])

        for col in X.columns:
            x = X[col]
            self.freq_maps[x.name] = x.value_counts().to_dict()

        return self

    def _fit_transform(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[pd.DataFrame, dict]:
        self._fit(X, y, **kwargs)
        X_out = self._transform(X)

        if self.keep_original:
            X_out = pd.concat([X, X_out], axis=1)

        # if self.log:
        #     type_group_map_special = {R_FLOAT: list(self.freq_maps.keys())}
        # else:
        #     type_group_map_special = {R_INT: list(self.freq_maps.keys())}
        return X_out, dict()  # TODO: Unsure whether we need to return anything special here

    def _transform(self, X_in, **kwargs):
        X = X_in.copy()

        new_cols = []
        for col in X.columns:
            x = X[col]
            if x.name in self.freq_maps:
                new_col = x.map(self.freq_maps[x.name]).astype(float).fillna(self.fillna)
                new_col.name = x.name + "_freq"
                if self.log:
                    new_col = np.log1p(new_col)
                    new_col.name += "_log"
                new_cols.append(new_col)
            else:
                continue
        if self.keep_original:
            return pd.concat([X] + new_cols, axis=1)
        else:
            return pd.concat([X[self.passthrough_cols]] + new_cols, axis=1)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(
            valid_raw_types=[R_OBJECT, R_CATEGORY, R_BOOL, R_INT, R_FLOAT],
            invalid_special_types=[S_DATETIME_AS_OBJECT, S_IMAGE_PATH, S_IMAGE_BYTEARRAY],
        )
