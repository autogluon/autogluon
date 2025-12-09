import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from typing import List, Dict, Tuple, Literal

from .abstract import AbstractFeatureGenerator
from autogluon.common.features.types import (
    R_BOOL,
    R_CATEGORY,
    R_OBJECT,
    R_INT,
    R_FLOAT,
    S_DATETIME_AS_OBJECT,
    S_IMAGE_BYTEARRAY,
    S_IMAGE_PATH,
    S_TEXT,
    S_TEXT_AS_CATEGORY,
)


class CatAsNumFeatureGenerator(AbstractFeatureGenerator):
    """
    Convert object/category columns:
      - If a column's non-null values are all numeric-like, cast it with pd.to_numeric.
      - Otherwise, apply an OrdinalEncoder (per-column) to map categories to integers.

    Parameters
    ----------
    target_type : str
        The type of the target variable ('regression', 'binary', or 'multiclass').
    keep_original : bool, default=False
        Whether to keep the original columns alongside the converted ones.
    handle_unknown : {"use_encoded_value", "error"}, default="use_encoded_value"
        Passed to each internal OrdinalEncoder.
    unknown_value : int or float, default=-1
        Value to use for unknown categories when handle_unknown="use_encoded_value".
    dtype : numpy dtype, default=np.float64
        Output dtype for numeric-like conversions and encoded columns.
    """

    def __init__(
        self,
        target_type: Literal["regression", "binary", "multiclass"],
        keep_original=False,
        handle_unknown: str = "use_encoded_value",
        unknown_value: float | int = -1,
        dtype=np.float64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_type = target_type
        self.keep_original = keep_original
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.dtype = dtype

    def estimate_no_of_new_features(self, X: pd.DataFrame, **kwargs) -> int:
        # TODO: Add dtype of new features to estimation for all preprocessors
        if self.keep_original:
            return X.select_dtypes(include=["object", "category"]).shape[1]
        else:  # TODO: Implement proper handling
            return X.select_dtypes(include=["object", "category"]).shape[1]

    def _fit(self, X_in: pd.DataFrame, y_in=None):
        X = X_in.copy()
        self.input_columns_: List[str] = list(X.columns)

        obj_cat_cols = X.select_dtypes(include=[R_OBJECT, R_CATEGORY]).columns.tolist()
        self.numeric_like_cols_: List[str] = []
        self.ordinal_cols_: List[str] = []
        self.passthrough_cols_: List[str] = [col for col in X.columns if col not in obj_cat_cols]
        self.encoders_: Dict[str, OrdinalEncoder] = {}

        # Decide column-by-column and fit encoders where needed
        for col in obj_cat_cols:
            s = X[col]
            # Determine if all non-null values can be parsed as numbers
            non_null = s.dropna()
            num_convertible = pd.to_numeric(non_null, errors="coerce").notna().all()

            if num_convertible:
                self.numeric_like_cols_.append(col)
            else:
                self.ordinal_cols_.append(col)
                enc = OrdinalEncoder(
                    handle_unknown=self.handle_unknown,
                    unknown_value=self.unknown_value,
                    dtype=self.dtype,
                )
                # Fit on a 2D array
                enc.fit(s.astype("category").to_frame())
                self.encoders_[col] = enc

        return self

    def _fit_transform(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[pd.DataFrame, dict]:
        self._fit(X, y, **kwargs)
        X_out = self._transform(X)
        return X_out, dict()

    def _transform(self, X_in: pd.DataFrame, **kwargs) -> pd.DataFrame:
        X_out = pd.DataFrame(index=X_in.index)

        # 1) Cast numeric-like string/category columns
        for col in self.numeric_like_cols_:
            if col in X_in.columns:
                X_out[col] = pd.to_numeric(X_in[col], errors="coerce").astype(self.dtype)

        # 2) Ordinal-encode the remaining categorical columns (per-column encoders)
        for col in self.ordinal_cols_:
            if col in X_in.columns:
                enc = self.encoders_[col]
                # transform expects 2D
                X_out[col] = enc.transform(X_in[[col]]).astype(self.dtype).ravel()

        X_out.columns = [f"{col}_num" for col in X_out.columns]
        if self.keep_original:
            X_out = pd.concat([X_in, X_out], axis=1)
        else:
            X_out = pd.concat([X_in[self.passthrough_cols_], X_out], axis=1)
        return X_out

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()
