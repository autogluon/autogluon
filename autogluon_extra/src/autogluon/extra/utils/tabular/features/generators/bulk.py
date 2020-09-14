import logging
from typing import List

import pandas as pd
from pandas import DataFrame

from .abstract import AbstractFeatureGenerator
from ..feature_metadata import FeatureMetadata

logger = logging.getLogger(__name__)


# TODO: Add parameter to add prefix to each generator to guarantee no name collisions: 'G1_', 'G2_', etc.
# TODO: Add argument keep_unused, which creates an identity feature generator at each stage to pipe unused input features into the next stage instead of dropping them.
class BulkFeatureGenerator(AbstractFeatureGenerator):
    """
    BulkFeatureGenerator is used for complex feature generation pipelines where multiple generators are required, with some generators requiring the output of other generators as input (multi-stage generation).
    For ML problems, it is expected that the user uses a feature generator that is an instance of or is inheriting from BulkFeatureGenerator, as single feature generators typically will not satisfy the feature generation needs of all input data types.
    Unless you are an expert user, we recommend you create custom FeatureGenerators based off of PipelineFeatureGenerator instead of BulkFeatureGenerator.

    Parameters
    ----------
    generators : List[List[AbstractFeatureGenerator]]
        generators is a list of generator groups, where a generator group is a list of generators.
        Feature generators within generators[i] (generator group) are all fit on the same data, and their outputs are then concatenated to form the output of generators[i].
        generators[i+1] are then fit on the output of generators[i].
        The last generator group's output is the output of _fit_transform and _transform methods.
        Due to the flexibility of generators, at the time of initialization, generators will prepend pre_generators and append post_generators if they are not None.
            If pre/post generators are specified, the supplied generators will be extended like this:
                pre_generators = [[pre_generator] for pre_generator in pre_generators]
                post_generators = [[post_generator] for post_generator in self._post_generators]
                self.generators: List[List[AbstractFeatureGenerator]] = pre_generators + generators + post_generators
                self._post_generators = []
            This means that self._post_generators will be empty as post_generators will be incorporated into self.generators instead.
        Note that if generators within a generator group produce a feature with the same name, an AssertionError will be raised as features with the same name cannot be present within a valid DataFrame output.
            If both features are desired, specify a name_prefix parameter in one of the generators to prevent name collisions.
            If experimenting with different generator groups, it is encouraged to try fitting your experimental feature-generators to the data without any ML model training to ensure validity and avoid name collisions.
    pre_generators: List[AbstractFeatureGenerator], optional
        pre_generators are generators which are sequentially fit prior to generators.
        Functions identically to post_generators argument, but pre_generators are called before generators, while post_generators are called after generators.
        Provided for convenience to classes inheriting from BulkFeatureGenerator.
        Common pre_generator's include AsTypeFeatureGenerator and FillNaFeatureGenerator, which act to prune and clean the data instead of generating entirely new features.
    **kwargs :
        Refer to AbstractFeatureGenerator documentation for details on valid key word arguments.

    Examples
    --------
    >>> from autogluon.extra import TabularPrediction as task
    >>> from autogluon.extra.utils.tabular.features.generators import AsTypeFeatureGenerator, BulkFeatureGenerator, CategoryFeatureGenerator, DropDuplicatesFeatureGenerator, FillNaFeatureGenerator, IdentityFeatureGenerator
    >>> from autogluon.extra.utils.tabular.features.feature_metadata import R_INT, R_FLOAT
    >>>
    >>> generators = [
    >>>     [AsTypeFeatureGenerator()],  # Convert all input features to the exact same types as they were during fit.
    >>>     [FillNaFeatureGenerator()],  # Fill all NA values in the data
    >>>     [
    >>>         CategoryFeatureGenerator(),  # Convert object types to category types and minimize their memory usage
    >>>         IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=[R_INT, R_FLOAT])),  # Carry over all features that are not objects and categories (without this, the int features would be dropped).
    >>>     ],  # CategoryFeatureGenerator and IdentityFeatureGenerator will have their outputs concatenated together before being fed into DropDuplicatesFeatureGenerator
    >>>     [DropDuplicatesFeatureGenerator()]  # Drops any features which are duplicates of each-other
    >>> ]
    >>> feature_generator = BulkFeatureGenerator(generators=generators, verbosity=3)
    >>>
    >>> label_column = 'class'
    >>> train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    >>> X_train = train_data.drop(labels=[label_column], axis=1)
    >>> y_train = train_data[label_column]
    >>>
    >>> X_train_transformed = feature_generator.fit_transform(X=X_train, y=y_train)
    >>>
    >>> test_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
    >>>
    >>> X_test_transformed = feature_generator.transform(test_data)
    """
    def __init__(self, generators: List[List[AbstractFeatureGenerator]], pre_generators: List[AbstractFeatureGenerator] = None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(generators, list):
            generators = [[generators]]
        elif len(generators) == 0:
            raise AssertionError('generators must contain at least one AbstractFeatureGenerator.')
        generators = [generator_group if isinstance(generator_group, list) else [generator_group] for generator_group in generators]
        if pre_generators is None:
            pre_generators = []
        elif not isinstance(pre_generators, list):
            pre_generators = [pre_generators]
        if self.pre_enforce_types:
            from .astype import AsTypeFeatureGenerator
            pre_generators = [AsTypeFeatureGenerator()] + pre_generators
            self.pre_enforce_types = False
        pre_generators = [[pre_generator] for pre_generator in pre_generators]

        if self._post_generators is not None:
            post_generators = [[post_generator] for post_generator in self._post_generators]
            self._post_generators = []
        else:
            post_generators = []
        self.generators: List[List[AbstractFeatureGenerator]] = pre_generators + generators + post_generators

        for generator_group in self.generators:
            for generator in generator_group:
                if not isinstance(generator, AbstractFeatureGenerator):
                    raise AssertionError(f'generators contains an object which is not an instance of AbstractFeatureGenerator. Invalid generator: {generator}')

        self._feature_metadata_in_unused: FeatureMetadata = None  # FeatureMetadata object based on the original input features that were unused by any feature generator.

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        feature_metadata = self.feature_metadata_in
        for i, generator_group in enumerate(self.generators):
            self._log(20, f'\tStage {i + 1} Generators:')
            feature_df_list = []
            for generator in generator_group:
                generator.set_log_prefix(log_prefix=self.log_prefix + '\t\t', prepend=True)
                feature_df_list.append(generator.fit_transform(X, feature_metadata_in=feature_metadata, **kwargs))

            self.generators[i] = [generator for j, generator in enumerate(generator_group) if feature_df_list[j] is not None and len(feature_df_list[j].columns) > 0]
            feature_df_list = [feature_df for feature_df in feature_df_list if feature_df is not None and len(feature_df.columns) > 0]

            if self.generators[i]:
                feature_metadata_in_generators = FeatureMetadata.join_metadatas([generator.feature_metadata_in for generator in self.generators[i]], shared_raw_features='error_if_diff')
            else:
                feature_metadata_in_generators = FeatureMetadata(type_map_raw=dict())

            if i == 0:  # TODO: Improve this
                used_features_in = feature_metadata_in_generators.get_features()
                unused_features_in = [feature for feature in self.feature_metadata_in.get_features() if feature not in used_features_in]
                self._feature_metadata_in_unused = self.feature_metadata_in.keep_features(features=unused_features_in)
                self._remove_features_in(features=unused_features_in)

            if self.generators[i]:
                feature_metadata = FeatureMetadata.join_metadatas([generator.feature_metadata for generator in self.generators[i]], shared_raw_features='error')
            else:
                feature_metadata = FeatureMetadata(type_map_raw=dict())

            if not feature_df_list:
                X = DataFrame(index=X.index)
            elif len(feature_df_list) == 1:
                X = feature_df_list[0]
            else:
                X = pd.concat(feature_df_list, axis=1, ignore_index=False, copy=False)
        X_out = X

        return X_out, feature_metadata.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        for generator_group in self.generators:
            feature_df_list = []
            for generator in generator_group:
                feature_df_list.append(generator.transform(X))

            if not feature_df_list:
                X = DataFrame(index=X.index)
            elif len(feature_df_list) == 1:
                X = feature_df_list[0]
            else:
                X = pd.concat(feature_df_list, axis=1, ignore_index=False, copy=False)
        X_out = X

        return X_out

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()
