import copy
import logging

import psutil
from pandas import DataFrame, Series

from .feature_metadata import FeatureMetadata
from .generators.dummy import DummyFeatureGenerator
from .generators.bulk import BulkFeatureGenerator
from .generators.fillna import FillNaFeatureGenerator
from .generators.drop_unique import DropUniqueFeatureGenerator
from .types import get_type_map_raw, get_type_map_real
from ..data.utils import get_approximate_df_mem_usage

logger = logging.getLogger(__name__)


# TODO: Add feature of # of observation counts to high cardinality categorical features
# TODO: Use code from problem type detection for column types. Ints/Floats could be Categorical through this method. Maybe try both?
class AbstractPipelineFeatureGenerator(BulkFeatureGenerator):
    def __init__(self, pre_generators=None, post_generators=None, pre_drop_useless=True, pre_enforce_types=True, reset_index=True, **kwargs):
        if pre_generators is None:
            pre_generators = [FillNaFeatureGenerator(inplace=True)]
        if post_generators is None:
            post_generators = [DropUniqueFeatureGenerator()]

        super().__init__(pre_generators=pre_generators, post_generators=post_generators, pre_drop_useless=pre_drop_useless, pre_enforce_types=pre_enforce_types, reset_index=reset_index, **kwargs)

        self._feature_metadata_in_real: FeatureMetadata = None  # FeatureMetadata object based on the original input features real dtypes (will contain dtypes such as 'int16' and 'float32' instead of 'int' and 'float').

        self._is_dummy = False  # If True, returns a single dummy feature as output. Occurs if fit with no useful features.

        self.pre_memory_usage = None
        self.pre_memory_usage_per_row = None
        self.post_memory_usage = None
        self.post_memory_usage_per_row = None

    def fit_transform(self, X: DataFrame, y=None, feature_metadata_in: FeatureMetadata = None, **kwargs) -> DataFrame:
        self._compute_pre_memory_usage(X)
        X_out = super().fit_transform(X=X, y=y, feature_metadata_in=feature_metadata_in, **kwargs)
        self._compute_post_memory_usage(X_out)

        logger.log(20, f'{self.__class__.__name__} processed %s data points with %s features' % (len(X_out), len(self.features_out)))
        self.print_feature_metadata_info()

        return X_out

    # TODO: Save this to disk and remove from memory if large categoricals!
    def _fit_transform(self, X: DataFrame, y=None, **kwargs):
        X_out, type_group_map_special = super()._fit_transform(X=X, y=y)
        X_out, type_group_map_special = self._fit_transform_custom(X_out=X_out, type_group_map_special=type_group_map_special, y=y)
        return X_out, type_group_map_special

    def _fit_transform_custom(self, X_out: DataFrame, type_group_map_special: dict, y=None) -> (DataFrame, dict):
        feature_metadata = FeatureMetadata(type_map_raw=get_type_map_raw(X_out), type_group_map_special=type_group_map_special)

        if len(list(X_out.columns)) == 0:
            self._is_dummy = True
            logger.warning(f'WARNING: No useful features were detected in the data! AutoGluon will train using 0 features, and will always predict the same value. Ensure that you are passing the correct data to AutoGluon!')
            dummy_generator = DummyFeatureGenerator()
            X_out = dummy_generator.fit_transform(X=X_out)
            type_group_map_special = copy.deepcopy(dummy_generator.feature_metadata.type_group_map_special)
            self.generators = [dummy_generator]
            self.post_generators = []
        else:
            type_group_map_special = feature_metadata.type_group_map_special

        return X_out, type_group_map_special

    def _infer_features_in_full(self, X: DataFrame, y: Series = None, feature_metadata_in: FeatureMetadata = None):
        super()._infer_features_in_full(X=X, y=y, feature_metadata_in=feature_metadata_in)
        type_map_real = get_type_map_real(X[self.feature_metadata_in.get_features()])
        self._feature_metadata_in_real = FeatureMetadata(type_map_raw=type_map_real, type_group_map_special=self.feature_metadata_in.type_group_map_raw)

    def _remove_features_in(self, features: list):
        super()._remove_features_in(features)
        if features:
            self._feature_metadata_in_real = self._feature_metadata_in_real.remove_features(features=features)

    def _compute_pre_memory_usage(self, X: DataFrame):
        X_len = len(X)
        self.pre_memory_usage = get_approximate_df_mem_usage(X, sample_ratio=0.2).sum()
        self.pre_memory_usage_per_row = self.pre_memory_usage / X_len
        available_mem = psutil.virtual_memory().available
        pre_memory_usage_percent = self.pre_memory_usage / (available_mem + self.pre_memory_usage)
        logger.log(20, f'Available Memory:                    {(round((self.pre_memory_usage + available_mem) / 1e6, 2))} MB')
        logger.log(20, f'Train Data (Original)  Memory Usage: {round(self.pre_memory_usage / 1e6, 2)} MB ({round(pre_memory_usage_percent * 100, 1)}% of available memory)')
        if pre_memory_usage_percent > 0.05:
            logger.warning(f'Warning: Data size prior to feature transformation consumes {round(pre_memory_usage_percent * 100, 1)}% of available memory. Consider increasing memory or subsampling the data to avoid instability.')

    def _compute_post_memory_usage(self, X: DataFrame):
        X_len = len(X)
        self.post_memory_usage = get_approximate_df_mem_usage(X, sample_ratio=0.2).sum()
        self.post_memory_usage_per_row = self.post_memory_usage / X_len

        available_mem = psutil.virtual_memory().available
        post_memory_usage_percent = self.post_memory_usage / (available_mem + self.post_memory_usage + self.pre_memory_usage)
        logger.log(20, f'Train Data (Processed) Memory Usage: {round(self.post_memory_usage / 1e6, 2)} MB ({round(post_memory_usage_percent * 100, 1)}% of available memory)')
        if post_memory_usage_percent > 0.15:
            logger.warning(f'Warning: Data size post feature transformation consumes {round(post_memory_usage_percent * 100, 1)}% of available memory. Consider increasing memory or subsampling the data to avoid instability.')

    def print_feature_metadata_info(self):
        if self._useless_features_in:
            logger.log(20, f'Useless Original Features (Count: {len(self._useless_features_in)}): {list(self._useless_features_in)}')
            logger.log(20, f'\tThese features carry no predictive signal and should be manually investigated.')  # TODO: What about features with 1 unique value but also np.nan?
            logger.log(20, f'\tThese features do not need to be present at inference time for this FeatureGenerator.')
        if self._feature_metadata_in_unused.get_features():
            logger.log(20, f'Unused Original Features (Count: {len(self._feature_metadata_in_unused.get_features())}): {self._feature_metadata_in_unused.get_features()}')
            logger.log(20, f'\tThese features were not valid input to any of the feature generators. Add a feature generator compatible with these features to utilize them.')
            logger.log(20, f'\tThese features do not need to be present at inference time for this FeatureGenerator.')
            self._feature_metadata_in_unused.print_feature_metadata_full('\t')
        logger.log(20, 'Original Features (exact raw dtype, raw dtype):')
        self._feature_metadata_in_real.print_feature_metadata_full('\t', print_only_one_special=True)
        super().print_feature_metadata_info()
