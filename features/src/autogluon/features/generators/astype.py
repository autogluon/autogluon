import logging

import numpy as np
import pandas as pd
from pandas import DataFrame

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.common.features.infer_types import get_bool_true_val, get_type_map_raw, get_type_map_real
from autogluon.common.features.types import R_INT, S_BOOL

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


# TODO: Add int fillna input value options: 0, set value, mean, mode, median
class AsTypeFeatureGenerator(AbstractFeatureGenerator):
    """
    Enforces type conversion on the data to match the types seen during fitting.
    If a feature cannot be converted to the correct type, an exception will be raised.

    Parameters
    ----------
    convert_bool : bool, default True
        Whether to automatically convert features with only two unique values to boolean.
    convert_bool_method : str, default "auto"
        [Advanced] The processing method to convert boolean features. Recommended to keep as "auto".
        If "auto": Will attempt to automatically select the best method based on the data.
        If "v1": Will use a simple method that was the default prior to v0.7 (`_convert_to_bool_simple`)
        If "v2": Will use an optimized method that was introduced in v0.7 (`_convert_to_bool_fast`)
        Note that "v2" is not always faster than "v1", and is often slower when there are few boolean columns.
        All options produce identical results, except in extreme synthetic edge-cases.
    convert_bool_method_v2_threshold : int, default 15
        [Advanced] If `convert_bool_method="auto"`, this value determines which method is used.
        If the number of boolean features is >= this value, then "v2" is used. Otherwise, "v1" is used.
        15 is roughly the optimal value on average.
    convert_bool_method_v2_row_threshold : int, default 128
        [Advanced] If using "v2" bool method, this is the row count in which when >=, the batch method is used instead of the realtime method.
        128 is roughly the optimal value on average.
    **kwargs :
        Refer to :class:`AbstractFeatureGenerator` documentation for details on valid key word arguments.
    """

    def __init__(
        self,
        convert_bool: bool = True,
        convert_bool_method: str = "auto",
        convert_bool_method_v2_threshold: int = 15,
        convert_bool_method_v2_row_threshold: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # FeatureMetadata object based on the original input features real dtypes
        # (will contain dtypes such as 'int16' and 'float32' instead of 'int' and 'float').
        self._feature_metadata_in_real: FeatureMetadata = None
        self._type_map_real_opt: dict = None  # Optimized representation of data types, saves a few milliseconds during comparisons in online inference
        # self.inplace = inplace  # TODO, also add check if dtypes are same as expected and skip .astype
        self._int_features = None
        self._bool_features = None
        self._convert_bool = convert_bool
        self._convert_bool_method_v2_threshold = convert_bool_method_v2_threshold
        self._convert_bool_method_v2_row_threshold = convert_bool_method_v2_row_threshold
        if convert_bool_method == "v1":
            self._use_fast_bool_method = False
        elif convert_bool_method == "v2":
            self._use_fast_bool_method = True
        elif convert_bool_method == "auto":
            self._use_fast_bool_method = "auto"
        else:
            raise ValueError(
                f"Unknown `convert_bool_method` value: {convert_bool_method}. " f'Valid values: ["v1", "v2", "auto"]'
            )
        self._bool_features_list = None
        self._non_bool_features_list = None
        self._bool_features_val = None
        self._bool_features_val_np = None

    # TODO: consider returning self._transform(X) if we allow users to specify real dtypes as input
    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        feature_type_raw_cur_dict = get_type_map_raw(X)
        feature_map_to_update = dict()
        type_map_special = self.feature_metadata_in.get_type_map_special()
        for feature in self.features_in:
            feature_type_raw = self.feature_metadata_in.get_feature_type_raw(feature)
            feature_type_raw_cur = feature_type_raw_cur_dict[feature]
            if feature_type_raw != feature_type_raw_cur:
                self._log(
                    30,
                    f'\tWARNING: Actual dtype differs from dtype in FeatureMetadata for feature "{feature}". '
                    f"Actual dtype: {feature_type_raw_cur} | Expected dtype: {feature_type_raw}",
                )
                feature_map_to_update[feature] = feature_type_raw
        if feature_map_to_update:
            self._log(
                30,
                "\tWARNING: Forcefully converting features to expected dtypes. "
                "Please manually align the input data with the expected dtypes if issues occur.",
            )
            X = X.astype(feature_map_to_update)

        self._bool_features = dict()
        if self._convert_bool:
            num_rows = len(X)
            if num_rows > 1000:
                # Sample and filter out features that already have >2 unique values
                # in the first 500 rows from bool consideration
                X_nunique_sample = X[self.features_in].head(500).nunique(dropna=False)
                X_nunique_sample = X_nunique_sample[X_nunique_sample <= 2]
                bool_candidates = list(X_nunique_sample.index)
            else:
                bool_candidates = self.features_in
            for feature in bool_candidates:
                if S_BOOL not in type_map_special[feature]:
                    uniques = X[feature].unique()
                    if len(uniques) == 2:
                        feature_bool_val = get_bool_true_val(uniques=uniques)
                        self._bool_features[feature] = feature_bool_val

        if self._bool_features:
            self._log(
                20,
                f"\tNote: Converting {len(self._bool_features)} features to boolean dtype "
                f"as they only contain 2 unique values.",
            )
            self._set_bool_features_val()
            if self._use_fast_bool_method == "auto":
                self._use_fast_bool_method = len(self._bool_features) >= self._convert_bool_method_v2_threshold
            X = self._convert_to_bool(X)
            for feature in self._bool_features:
                type_map_special[feature] = [S_BOOL]
                self._type_map_real_opt[feature] = np.int8
            type_group_map_special = FeatureMetadata.get_type_group_map_special_from_type_map_special(type_map_special)
        else:
            type_group_map_special = self.feature_metadata_in.type_group_map_special
        self._int_features = np.array(self.feature_metadata_in.get_features(valid_raw_types=[R_INT]))
        return X, type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        if self._bool_features:
            X = self._convert_to_bool(X)
        # check if not same
        if self._type_map_real_opt != X.dtypes.to_dict():
            if self._int_features.size:
                null_count = X[self._int_features].isnull().any()
                # If int feature contains null during inference but not during fit.
                if null_count.any():
                    # TODO: Consider imputing to mode? This is tricky because training data had no missing values.
                    # TODO: Add unit test for this situation, to confirm it is handled properly.
                    with_null = null_count[null_count]
                    with_null_features = list(with_null.index)
                    logger.warning(
                        "WARNING: Int features without null values "
                        "at train time contain null values at inference time! "
                        "Imputing nulls to 0. To avoid this, pass the features as floats during fit!"
                    )
                    logger.warning(f"WARNING: Int features with nulls: {with_null_features}")
                    X[with_null_features] = X[with_null_features].fillna(0)

            if self._type_map_real_opt:
                # TODO: Confirm this works with sparse and other feature types!
                # FIXME: Address situation where test-time invalid type values cause crash:
                #  https://stackoverflow.com/questions/49256211/how-to-set-unexpected-data-type-to-na?noredirect=1&lq=1
                try:
                    X = X.astype(self._type_map_real_opt)
                except Exception as e:
                    self._log_invalid_dtypes(X=X)
                    raise e
        return X

    def _log_invalid_dtypes(self, X: pd.DataFrame):
        """
        Logs detailed information on all feature transformations, including exceptions that occur.
        """
        pd_cols = ["feature", "status", "dtype_input", "dtype_to_convert_to", "exception"]
        rows = []

        logger.log(
            40,
            f"Exception encountered in {self.__class__.__name__} ... "
            f"Please check if feature data types differ between train and test (via df.dtypes).\nException breakdown by feature:",
        )
        for f in self._type_map_real_opt.keys():
            f_type_out = self._type_map_real_opt[f]
            f_type_in = X[f].dtype
            try:
                X[f].astype(f_type_out)
            except Exception as e:
                status = f"{e.__class__.__name__}"
                exception = e
            else:
                status = "Success"
                exception = None
            row = [f, status, f_type_in, f_type_out, exception]
            rows.append(row)
        df_debug = pd.DataFrame(rows, columns=pd_cols)
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            logger.log(40, df_debug)

    def _convert_to_bool(self, X: DataFrame) -> DataFrame:
        if self._use_fast_bool_method:
            return self._convert_to_bool_fast(X)
        else:
            return self._convert_to_bool_simple(X)

    def _convert_to_bool_simple(self, X: DataFrame) -> DataFrame:
        """Generic method to convert feature types to booleans. Efficient with small amounts of features."""
        for feature in self._bool_features_list:
            # Note, this edits inplace, altering outer context.
            #  This is ok when used in PipelineFeatureGenerator, as the data is already deep copied.
            #  We avoid deep copying here to speed up processing.
            X[feature] = (X[feature] == self._bool_features[feature]).astype(np.int8)
        return X

    def _convert_to_bool_fast(self, X: DataFrame) -> DataFrame:
        """
        Faster method to convert feature types to boolean when many features must be converted at once.
        Can be >10x faster than the simple version, particularly when len(X) < 100

        Note that the fast method alters the column order with boolean features being last.
        """
        if len(X) >= self._convert_bool_method_v2_row_threshold:
            return self._convert_to_bool_fast_batch(X)
        else:
            return self._convert_to_bool_fast_realtime(X)

    def _convert_to_bool_fast_batch(self, X: DataFrame) -> DataFrame:
        """Optimized for when X is > 100 rows"""
        X_bool_list = []
        for feature in self._bool_features_list:
            X_bool_list.append((X[feature] == self._bool_features[feature]).astype(np.int8))
        X_bool = pd.concat(X_bool_list, axis=1)

        # TODO: re-order columns to features_in required because `feature_interactions=False` to avoid error when feature prune.
        #  Note that this is slower than avoiding the re-order, but avoiding the re-order is very complicated to do correctly.
        return pd.concat([X[self._non_bool_features_list], X_bool], axis=1)[self.features_in]

    def _convert_to_bool_fast_realtime(self, X: DataFrame) -> DataFrame:
        """Optimized for when X is <= 100 rows"""
        X_bool_features_np = X[self._bool_features_list].to_numpy(dtype="object")
        X_bool_numpy = X_bool_features_np == self._bool_features_val_np
        X_bool = pd.DataFrame(X_bool_numpy, columns=self._bool_features_list, dtype=np.int8, index=X.index)

        # TODO: re-order columns to features_in required because `feature_interactions=False` to avoid error when feature prune.
        #  Note that this is slower than avoiding the re-order, but avoiding the re-order is very complicated to do correctly.
        return pd.concat([X[self._non_bool_features_list], X_bool], axis=1)[self.features_in]

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()

    def _infer_features_in_full(self, X: DataFrame, feature_metadata_in: FeatureMetadata = None):
        super()._infer_features_in_full(X=X, feature_metadata_in=feature_metadata_in)
        type_map_real = get_type_map_real(X[self.feature_metadata_in.get_features()])
        self._type_map_real_opt = X[self.feature_metadata_in.get_features()].dtypes.to_dict()
        self._feature_metadata_in_real = FeatureMetadata(
            type_map_raw=type_map_real, type_group_map_special=self.feature_metadata_in.get_type_group_map_raw()
        )

    def _remove_features_in(self, features):
        super()._remove_features_in(features)
        if features:
            self._feature_metadata_in_real = self._feature_metadata_in_real.remove_features(features=features)
            for feature in features:
                self._type_map_real_opt.pop(feature, None)
                self._bool_features.pop(feature, None)
            self._set_bool_features_val()
            self._int_features = np.array(self.feature_metadata_in.get_features(valid_raw_types=[R_INT]))

    def _set_bool_features_val(self):
        self._bool_features_val = [self._bool_features[f] for f in self._bool_features]
        self._bool_features_val_np = np.array(self._bool_features_val, dtype="object")
        self._bool_features_list = list(self._bool_features.keys())
        self._non_bool_features_list = [f for f in self.features_in if f not in self._bool_features]

    def print_feature_metadata_info(self, log_level=20):
        self._log(log_level, "\tOriginal Features (exact raw dtype, raw dtype):")
        self._feature_metadata_in_real.print_feature_metadata_full(
            self.log_prefix + "\t\t", print_only_one_special=True, log_level=log_level
        )
        super().print_feature_metadata_info(log_level=log_level)

    def _more_tags(self):
        return {"feature_interactions": False}
