import copy
import logging

import numpy as np
from pandas import DataFrame

from autogluon.core.features.types import R_INT, S_BOOL
from autogluon.core.features.feature_metadata import FeatureMetadata
from autogluon.core.features.infer_types import get_type_map_raw, get_type_map_real, get_bool_true_val

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
    **kwargs :
        Refer to :class:`AbstractFeatureGenerator` documentation for details on valid key word arguments.
    """
    def __init__(self, convert_bool=True, **kwargs):
        super().__init__(**kwargs)
        self._feature_metadata_in_real: FeatureMetadata = None  # FeatureMetadata object based on the original input features real dtypes (will contain dtypes such as 'int16' and 'float32' instead of 'int' and 'float').
        self._type_map_real_opt: dict = None  # Optimized representation of data types, saves a few milliseconds during comparisons in online inference
        # self.inplace = inplace  # TODO, also add check if dtypes are same as expected and skip .astype
        self._int_features = None
        self._bool_features = None
        self._convert_bool = convert_bool

    # TODO: consider returning self._transform(X) if we allow users to specify real dtypes as input
    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        feature_type_raw_cur_dict = get_type_map_raw(X)
        feature_map_to_update = dict()
        type_map_special = self.feature_metadata_in.get_type_map_special()
        for feature in self.features_in:
            feature_type_raw = self.feature_metadata_in.get_feature_type_raw(feature)
            feature_type_raw_cur = feature_type_raw_cur_dict[feature]
            if feature_type_raw != feature_type_raw_cur:
                self._log(30, f'\tWARNING: Actual dtype differs from dtype in FeatureMetadata for feature "{feature}". Actual dtype: {feature_type_raw_cur} | Expected dtype: {feature_type_raw}')
                feature_map_to_update[feature] = feature_type_raw
        if feature_map_to_update:
            self._log(30, f'\tWARNING: Forcefully converting features to expected dtypes. Please manually align the input data with the expected dtypes if issues occur.')
            X = X.astype(feature_map_to_update)

        self._bool_features = dict()
        if self._convert_bool:
            for feature in self.features_in:
                if S_BOOL not in type_map_special[feature]:
                    if len(X[feature].unique()) == 2:
                        feature_bool_val = get_bool_true_val(X[feature])
                        self._bool_features[feature] = feature_bool_val

        if self._bool_features:
            self._log(20, f'\tNote: Converting {len(self._bool_features)} features to boolean dtype as they only contain 2 unique values.')
            for feature in self._bool_features:
                type_map_special[feature] = [S_BOOL]
                X[feature] = (X[feature] == self._bool_features[feature]).astype(np.int8)
                self._type_map_real_opt[feature] = np.int8
            type_group_map_special = FeatureMetadata.get_type_group_map_special_from_type_map_special(type_map_special)
        else:
            type_group_map_special = self.feature_metadata_in.type_group_map_special
        self._int_features = np.array(self.feature_metadata_in.get_features(valid_raw_types=[R_INT]))
        return X, type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        if self._bool_features:
            for feature in self._bool_features:
                X[feature] = (X[feature] == self._bool_features[feature]).astype(np.int8)
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
                    logger.warning(f'WARNING: Int features without null values at train time contain null values at inference time! Imputing nulls to 0. To avoid this, pass the features as floats during fit!')
                    logger.warning(f'WARNING: Int features with nulls: {with_null_features}')
                    X[with_null_features] = X[with_null_features].fillna(0)

            if self._type_map_real_opt:
                # TODO: Confirm this works with sparse and other feature types!
                # FIXME: Address situation where test-time invalid type values cause crash:
                #  https://stackoverflow.com/questions/49256211/how-to-set-unexpected-data-type-to-na?noredirect=1&lq=1
                X = X.astype(self._type_map_real_opt)
        return X

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()

    def _infer_features_in_full(self, X: DataFrame, feature_metadata_in: FeatureMetadata = None):
        super()._infer_features_in_full(X=X, feature_metadata_in=feature_metadata_in)
        type_map_real = get_type_map_real(X[self.feature_metadata_in.get_features()])
        self._type_map_real_opt = X[self.feature_metadata_in.get_features()].dtypes.to_dict()
        self._feature_metadata_in_real = FeatureMetadata(type_map_raw=type_map_real, type_group_map_special=self.feature_metadata_in.get_type_group_map_raw())

    def _remove_features_in(self, features):
        super()._remove_features_in(features)
        if features:
            self._feature_metadata_in_real = self._feature_metadata_in_real.remove_features(features=features)
            for feature in features:
                self._type_map_real_opt.pop(feature, None)
                self._bool_features.pop(feature, None)
            self._int_features = np.array(self.feature_metadata_in.get_features(valid_raw_types=[R_INT]))

    def print_feature_metadata_info(self, log_level=20):
        self._log(log_level, '\tOriginal Features (exact raw dtype, raw dtype):')
        self._feature_metadata_in_real.print_feature_metadata_full(self.log_prefix + '\t\t', print_only_one_special=True, log_level=log_level)
        super().print_feature_metadata_info(log_level=log_level)

    def _more_tags(self):
        return {'feature_interactions': False}
