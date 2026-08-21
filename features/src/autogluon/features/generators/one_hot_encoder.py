import logging
import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder

from autogluon.common.features.types import R_CATEGORY, R_INT, S_BOOL, S_SPARSE

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


class CatToInt:
    """
    Converts pandas categoricals to numpy int in preparation for OHE.

    Parameters
    ----------
    max_levels : int
        The maximum number of unique values for OHE. Selected categories are based on frequency.
    fillna_val : int, default = None
        The default value to fill NaN.
        If None, automatically inferred as a new value not present in existing categories.
    infrequent_val : int or {'na', 'na+1'}, default = 'na'
        The value to group all infrequent categories to (those that aren't within the max_levels most frequent categories).
        If 'na', uses `fillna_val`.
        If 'na+1', uses `fillna_val+1`. This guarantees a new category for infrequent values separate from missing values if `fillna_val=None`.
    """

    def __init__(self, max_levels, fillna_val=None, infrequent_val="na"):
        self.max_levels = max_levels
        self.fillna_val = fillna_val
        self.infrequent_val = infrequent_val
        self.cats = dict()
        self.num_cols = None
        self._dtype = None

    def fit(self, X: DataFrame):
        # dtype_buffer=2 is required to avoid edge case errors with invalid self.infrequent_val in 'na+1' mode.
        self._dtype, fillna_val = self._get_dtype_and_fillna(X, dtype_buffer=2)
        if self.fillna_val is None:
            self.fillna_val = fillna_val
        if self.infrequent_val == "na":
            self.infrequent_val = self.fillna_val
        elif self.infrequent_val == "na+1":
            self.infrequent_val = self.fillna_val + 1

        X = self.pd_to_np(X)
        self.num_cols = X.shape[1]
        for col in range(self.num_cols):
            data = X[:, col]
            uniques, counts = np.unique(data, return_counts=True)
            self.cats[col] = uniques[np.argsort(counts)[-self.max_levels :]]
            with warnings.catch_warnings():
                # Refer to https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
                warnings.simplefilter(action="ignore", category=FutureWarning)
                if self.infrequent_val in self.cats[col] or str(self.infrequent_val) in self.cats[col]:
                    # Add one extra level since NaN values shouldn't be counted towards max levels
                    self.cats[col] = uniques[np.argsort(counts)[-(self.max_levels + 1) :]]

    def transform(self, X: DataFrame):
        X = self.pd_to_np(X)
        mask = np.zeros(shape=X.shape, dtype=bool)
        for col in range(self.num_cols):
            mask[:, col] = np.isin(X[:, col], self.cats[col], invert=True)
        X[mask] = self.infrequent_val
        return X

    def pd_to_np(self, X: DataFrame) -> np.ndarray:
        """
        Converts pandas categoricals to a numpy ndarray of the codes of the categories.
        """
        with warnings.catch_warnings():
            if np.issubdtype(self._dtype, np.integer):
                # Filter incorrect pandas RuntimeWarning message
                # For more details, refer to https://github.com/autogluon/autogluon/pull/4224#issuecomment-2156423410
                warnings.filterwarnings("ignore", category=RuntimeWarning)
            return X.to_numpy(dtype=self._dtype, na_value=self.fillna_val, copy=True)

    def _get_dtype_and_fillna(self, X: DataFrame, dtype_buffer=2):
        assert dtype_buffer >= 1, "dtype_buffer must be >= 1 or else fillna_val could be invalid."
        dtype = None
        max_val_all = None
        for col in X.columns:
            try:
                max_val = X[col].dtype.categories.max()
                min_val = X[col].dtype.categories.min()
            except:
                max_val = X[col].max()
                min_val = X[col].min()
            if isinstance(max_val, str):
                max_dtype = np.min_scalar_type(max_val)
            else:
                if max_val_all is None:
                    max_val_all = max_val
                else:
                    max_val_all = max(max_val_all, max_val)
                max_val_with_buffer = max_val + dtype_buffer
                max_dtype = np.min_scalar_type(max_val_with_buffer)
            min_dtype = np.min_scalar_type(min_val)
            cur_dtype = np.promote_types(min_dtype, max_dtype)

            if dtype is None:
                dtype = cur_dtype
            else:
                dtype = np.promote_types(dtype, cur_dtype)
        if max_val_all is None:
            fillna_val = 0
        else:
            fillna_val = max_val_all + 1
        return dtype, fillna_val


# TODO: Replace XGBoost, NN, and Linear Model OHE logic with this
class OneHotEncoderFeatureGenerator(AbstractFeatureGenerator):
    """
    Converts category features to one-hot boolean features by mapping to the category codes.

    Parameters
    ----------
    max_levels : int
        The maximum number of categories to use for OHE per feature. Selected categories are based on frequency.
    dtype : number type, default = np.uint8
        Desired dtype of output.
    sparse : bool, default = True
        Will return sparse matrix if set to True else will return an array.
    drop : str, default = None
        Refer to OneHotEncoder documentation for details.
    """

    def __init__(self, max_levels=None, dtype=np.uint8, sparse=True, drop=None, **kwargs):
        super().__init__(**kwargs)
        self.max_levels = max_levels
        self.sparse = sparse
        self._ohe = OneHotEncoder(dtype=dtype, sparse_output=self.sparse, handle_unknown="ignore", drop=drop)
        self._ohe_columns = None
        self._cat_feat_gen = None

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        if self.max_levels is not None:
            self._cat_feat_gen = CatToInt(max_levels=self.max_levels)
            self._cat_feat_gen.fit(X)
            X_out = self._cat_feat_gen.transform(X)
        else:
            X_out = X

        self._ohe.fit(X_out)
        self._ohe_columns = self._ohe.get_feature_names_out()
        self._ohe_columns = ["_ohe_" + str(i) for i in range(len(self._ohe_columns))]
        X_out = self._transform(X)

        features_out = list(X_out.columns)

        feature_metadata_out_type_group_map_special = {S_BOOL: features_out}
        if self.sparse:
            feature_metadata_out_type_group_map_special[S_SPARSE] = features_out
        return X_out, feature_metadata_out_type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        X_out = self.transform_ohe(X)
        if self.sparse:
            X_out = pd.DataFrame.sparse.from_spmatrix(X_out, columns=self._ohe_columns, index=X.index)
        else:
            X_out = pd.DataFrame(X_out, columns=self._ohe_columns, index=X.index)
        return X_out

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_CATEGORY, R_INT])

    def transform_ohe(self, X: DataFrame):
        """
        Call this method directly to get numpy output.
        Skips pandas conversion (much faster if only the numpy output is required).
        """
        if self._cat_feat_gen is not None:
            X = self._cat_feat_gen.transform(X)
        X = self._ohe.transform(X)
        return X

    def _more_tags(self):
        return {"feature_interactions": False}
