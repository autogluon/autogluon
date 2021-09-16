import copy
import logging

import psutil
from pandas import DataFrame

from autogluon.core.features.feature_metadata import FeatureMetadata
from autogluon.core.features.infer_types import get_type_map_real
from autogluon.core.utils import get_approximate_df_mem_usage

from .bulk import BulkFeatureGenerator
from .dummy import DummyFeatureGenerator
from .drop_unique import DropUniqueFeatureGenerator
from .fillna import FillNaFeatureGenerator

logger = logging.getLogger(__name__)


# TODO: Documentation
class PipelineFeatureGenerator(BulkFeatureGenerator):
    """
    PipelineFeatureGenerator is an implementation of BulkFeatureGenerator with various smart defaults and edge case handling functionality to enable robust data handling.
    It is recommended that users base any custom feature generators meant for end-to-end data transformation from PipelineFeatureGenerator.
        Reference AutoMLPipelineFeatureGenerator for an example of extending PipelineFeatureGenerator.
    It is not recommended that PipelineFeatureGenerator be used as a generator within any other generator's pre or post generators.
    """
    def __init__(self, pre_generators=None, post_generators=None, pre_drop_useless=True, pre_enforce_types=True, reset_index=True, verbosity=3, **kwargs):
        if pre_generators is None:
            pre_generators = [FillNaFeatureGenerator(inplace=True)]
        if post_generators is None:
            post_generators = [DropUniqueFeatureGenerator()]

        super().__init__(pre_generators=pre_generators, post_generators=post_generators, pre_drop_useless=pre_drop_useless, pre_enforce_types=pre_enforce_types, reset_index=reset_index, verbosity=verbosity, **kwargs)

        self._feature_metadata_in_real: FeatureMetadata = None  # FeatureMetadata object based on the original input features real dtypes (will contain dtypes such as 'int16' and 'float32' instead of 'int' and 'float').

        self._is_dummy = False  # If True, returns a single dummy feature as output. Occurs if fit with no useful features.

        self.pre_memory_usage = None
        self.pre_memory_usage_per_row = None
        self.post_memory_usage = None
        self.post_memory_usage_per_row = None

    def fit_transform(self, X: DataFrame, y=None, feature_metadata_in: FeatureMetadata = None, **kwargs) -> DataFrame:
        X_out = super().fit_transform(X=X, y=y, feature_metadata_in=feature_metadata_in, **kwargs)
        self._compute_post_memory_usage(X_out)
        # TODO: Consider adding final check of validity/that features are reasonable.

        return X_out

    def _fit_transform(self, X: DataFrame, y=None, **kwargs):
        X_out, type_group_map_special = super()._fit_transform(X=X, y=y, **kwargs)
        X_out, type_group_map_special = self._fit_transform_custom(X_out=X_out, type_group_map_special=type_group_map_special, y=y)
        return X_out, type_group_map_special

    def _fit_transform_custom(self, X_out: DataFrame, type_group_map_special: dict, y=None) -> (DataFrame, dict):
        if len(list(X_out.columns)) == 0:
            self._is_dummy = True
            self._log(30, f'\tWARNING: No useful features were detected in the data! AutoGluon will train using 0 features, and will always predict the same value. Ensure that you are passing the correct data to AutoGluon!')
            dummy_generator = DummyFeatureGenerator()
            X_out = dummy_generator.fit_transform(X=X_out)
            type_group_map_special = copy.deepcopy(dummy_generator.feature_metadata.type_group_map_special)
            self.generators = [[dummy_generator]]
            self._remove_features_in(features=self.features_in)
        return X_out, type_group_map_special

    def _infer_features_in_full(self, X: DataFrame, feature_metadata_in: FeatureMetadata = None):
        super()._infer_features_in_full(X=X, feature_metadata_in=feature_metadata_in)
        type_map_real = get_type_map_real(X[self.feature_metadata_in.get_features()])
        self._feature_metadata_in_real = FeatureMetadata(type_map_raw=type_map_real, type_group_map_special=self.feature_metadata_in.get_type_group_map_raw())

    def _remove_features_in(self, features: list):
        super()._remove_features_in(features)
        if features:
            self._feature_metadata_in_real = self._feature_metadata_in_real.remove_features(features=features)

    def _pre_fit_validate(self, X: DataFrame, **kwargs):
        super()._pre_fit_validate(X=X, **kwargs)
        self._ensure_no_duplicate_column_names(X=X)  # TODO: Remove this, move pre_memory_usage and post_memory_usage into super().
        self._compute_pre_memory_usage(X)

    def _compute_pre_memory_usage(self, X: DataFrame):
        X_len = len(X)
        self.pre_memory_usage = get_approximate_df_mem_usage(X, sample_ratio=0.2).sum()
        self.pre_memory_usage_per_row = self.pre_memory_usage / X_len
        available_mem = psutil.virtual_memory().available
        pre_memory_usage_percent = self.pre_memory_usage / (available_mem + self.pre_memory_usage)
        self._log(20, f'\tAvailable Memory:                    {(round((self.pre_memory_usage + available_mem) / 1e6, 2))} MB')
        self._log(20, f'\tTrain Data (Original)  Memory Usage: {round(self.pre_memory_usage / 1e6, 2)} MB ({round(pre_memory_usage_percent * 100, 1)}% of available memory)')
        if pre_memory_usage_percent > 0.05:
            self._log(30, f'\tWarning: Data size prior to feature transformation consumes {round(pre_memory_usage_percent * 100, 1)}% of available memory. Consider increasing memory or subsampling the data to avoid instability.')

    def _compute_post_memory_usage(self, X: DataFrame):
        X_len = len(X)
        self.post_memory_usage = get_approximate_df_mem_usage(X, sample_ratio=0.2).sum()
        self.post_memory_usage_per_row = self.post_memory_usage / X_len

        available_mem = psutil.virtual_memory().available
        post_memory_usage_percent = self.post_memory_usage / (available_mem + self.post_memory_usage + self.pre_memory_usage)
        self._log(20, f'\tTrain Data (Processed) Memory Usage: {round(self.post_memory_usage / 1e6, 2)} MB ({round(post_memory_usage_percent * 100, 1)}% of available memory)')
        if post_memory_usage_percent > 0.15:
            self._log(30, f'\tWarning: Data size post feature transformation consumes {round(post_memory_usage_percent * 100, 1)}% of available memory. Consider increasing memory or subsampling the data to avoid instability.')

    def print_feature_metadata_info(self, log_level=20):
        if self._useless_features_in:
            self._log(log_level, f'\tUseless Original Features (Count: {len(self._useless_features_in)}): {list(self._useless_features_in)}')
            self._log(log_level, f'\t\tThese features carry no predictive signal and should be manually investigated.')  # TODO: What about features with 1 unique value but also np.nan?
            self._log(log_level, f'\t\tThis is typically a feature which has the same value for all rows.')
            self._log(log_level, f'\t\tThese features do not need to be present at inference time.')
        if self._feature_metadata_in_unused.get_features():
            # TODO: Consider highlighting why a feature was unused (complex to implement, can check if was valid input to any generator in a generator group through feature chaining)
            self._log(log_level, f'\tUnused Original Features (Count: {len(self._feature_metadata_in_unused.get_features())}): {self._feature_metadata_in_unused.get_features()}')
            self._log(log_level, f'\t\tThese features were not used to generate any of the output features. Add a feature generator compatible with these features to utilize them.')
            self._log(log_level, f'\t\tFeatures can also be unused if they carry very little information, such as being categorical but having almost entirely unique values or being duplicates of other features.')
            self._log(log_level, f'\t\tThese features do not need to be present at inference time.')
            self._feature_metadata_in_unused.print_feature_metadata_full(self.log_prefix + '\t\t', log_level=log_level)
        self._log(log_level-5, '\tTypes of features in original data (exact raw dtype, raw dtype):')
        self._feature_metadata_in_real.print_feature_metadata_full(self.log_prefix + '\t\t', print_only_one_special=True, log_level=log_level-5)
        super().print_feature_metadata_info(log_level=log_level)
