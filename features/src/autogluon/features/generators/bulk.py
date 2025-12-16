from __future__ import annotations

import logging
from typing import List

from pandas import DataFrame

from autogluon.common.features.feature_metadata import FeatureMetadata

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


# TODO: Add parameter to add prefix to each generator to guarantee no name collisions: 'G1_', 'G2_', etc.
# TODO: Add argument keep_unused, which creates an identity feature generator at each stage to pipe unused
#  input features into the next stage instead of dropping them.
class BulkFeatureGenerator(AbstractFeatureGenerator):
    """
    BulkFeatureGenerator is used for complex feature generation pipelines where multiple generators are required,
    with some generators requiring the output of other generators as input (multi-stage generation).
    For ML problems, it is expected that the user uses a feature generator that is an instance of or is inheriting from BulkFeatureGenerator,
    as single feature generators typically will not satisfy the feature generation needs of all input data types.
    Unless you are an expert user, we recommend you create custom FeatureGenerators based off of PipelineFeatureGenerator instead of BulkFeatureGenerator.

    Parameters
    ----------
    generators : List[List[:class:`AbstractFeatureGenerator`]]
        generators is a list of generator groups, where a generator group is a list of generators.
        Feature generators within generators[i] (generator group) are all fit on the same data,
        and their outputs are then concatenated to form the output of generators[i].
        generators[i+1] are then fit on the output of generators[i].
        The last generator group's output is the output of _fit_transform and _transform methods.
        Due to the flexibility of generators, at the time of initialization, generators will prepend pre_generators and append post_generators
        if they are not None.
            If pre/post generators are specified, the supplied generators will be extended like this:
                pre_generators = [[pre_generator] for pre_generator in pre_generators]
                post_generators = [[post_generator] for post_generator in self._post_generators]
                self.generators: List[List[AbstractFeatureGenerator]] = pre_generators + generators + post_generators
                self._post_generators = []
            This means that self._post_generators will be empty as post_generators will be incorporated into self.generators instead.
        Note that if generators within a generator group produce a feature with the same name, an AssertionError will be raised as features
        with the same name cannot be present within a valid DataFrame output.
            If both features are desired, specify a name_prefix parameter in one of the generators to prevent name collisions.
            If experimenting with different generator groups, it is encouraged to try fitting your experimental
            feature-generators to the data without any ML model training to ensure validity and avoid name collisions.
    pre_generators: List[AbstractFeatureGenerator], optional
        pre_generators are generators which are sequentially fit prior to generators.
        Functions identically to post_generators argument, but pre_generators are called before generators, while post_generators are called after generators.
        Provided for convenience to classes inheriting from BulkFeatureGenerator.
        Common pre_generator's include :class:`AsTypeFeatureGenerator` and :class:`FillNaFeatureGenerator`, which act to prune and clean the data instead
        of generating entirely new features.
    remove_unused_features: {True, False, "false_recursive"}, default True
        If True, will try to remove unused input features based on the output features post-fit.
        This is done to optimize inference speed.
        If False, will not perform this operation.
        If "false_recursive", will also disable this operation in all inner generators.
    **kwargs :
        Refer to :class:`AbstractFeatureGenerator` documentation for details on valid key word arguments.

    Examples
    --------
    >>> from autogluon.tabular import TabularDataset
    >>> from autogluon.features.generators import AsTypeFeatureGenerator, BulkFeatureGenerator, CategoryFeatureGenerator, DropDuplicatesFeatureGenerator, FillNaFeatureGenerator, IdentityFeatureGenerator  # noqa
    >>> from autogluon.common.features.types import R_INT, R_FLOAT
    >>>
    >>> generators = [
    >>>     [AsTypeFeatureGenerator()],  # Convert all input features to the exact same types as they were during fit.
    >>>     [FillNaFeatureGenerator()],  # Fill all NA values in the data
    >>>     [
    >>>         CategoryFeatureGenerator(),  # Convert object types to category types and minimize their memory usage
    >>>         # Carry over all features that are not objects and categories (without this, the int features would be dropped).
    >>>         IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=[R_INT, R_FLOAT])),
    >>>     ],
    >>>     # CategoryFeatureGenerator and IdentityFeatureGenerator will have their outputs concatenated together
    >>>     # before being fed into DropDuplicatesFeatureGenerator
    >>>     [DropDuplicatesFeatureGenerator()]  # Drops any features which are duplicates of each-other
    >>> ]
    >>> feature_generator = BulkFeatureGenerator(generators=generators, verbosity=3)
    >>>
    >>> label = 'class'
    >>> train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    >>> X_train = train_data.drop(labels=[label], axis=1)
    >>> y_train = train_data[label]
    >>>
    >>> X_train_transformed = feature_generator.fit_transform(X=X_train, y=y_train)
    >>>
    >>> test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
    >>>
    >>> X_test_transformed = feature_generator.transform(test_data)
    """

    def __init__(
        self,
        generators: list[list[AbstractFeatureGenerator | list]],
        pre_generators: list[AbstractFeatureGenerator] = None,
        remove_unused_features: bool | str = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(remove_unused_features, str):
            assert remove_unused_features == "false_recursive", (
                "remove_unused_features only accepts bool or 'false_recursive'"
            )
            self._remove_unused_features_flag = False
        else:
            assert isinstance(remove_unused_features, bool)
            self._remove_unused_features_flag = remove_unused_features
        if not isinstance(generators, list):
            generators = [[generators]]
        elif len(generators) == 0:
            raise AssertionError("generators must contain at least one AbstractFeatureGenerator.")
        _generators_init = [
            generator_group if isinstance(generator_group, list) else [generator_group]
            for generator_group in generators
        ]
        generators = []
        for generator_group in _generators_init:
            _generators_group = []
            for generator_group_inner in generator_group:
                if isinstance(generator_group_inner, list):
                    inner_Kwargs = {}
                    if isinstance(remove_unused_features, str) and remove_unused_features == "false_recursive":
                        inner_Kwargs["remove_unused_features"] = remove_unused_features
                    _generators_group.append(
                        BulkFeatureGenerator(
                            generators=generator_group_inner,
                            verbosity=self.verbosity,
                            **inner_Kwargs,
                        )
                    )
                else:
                    _generators_group.append(generator_group_inner)
            generators.append(_generators_group)
        _generators_init = None

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
                    raise AssertionError(
                        f"generators contains an object which is not an instance of AbstractFeatureGenerator. Invalid generator: {generator}"
                    )

        # FeatureMetadata object based on the original input features that were unused by any feature generator.
        self._feature_metadata_in_unused: FeatureMetadata = None

    def _fit_transform(self, X: DataFrame, **kwargs) -> tuple[DataFrame, dict]:
        feature_metadata = self.feature_metadata_in
        for i in range(len(self.generators)):
            self._log(20, f"\tStage {i + 1} Generators:")
            X, self.generators[i], feature_metadata = self._fit_transform_stage(
                X=X,
                generators=self.generators[i],
                feature_metadata_in=feature_metadata,
                **kwargs,
            )

        if self._remove_unused_features_flag:
            self._remove_features_out(features=[])
        # Remove useless generators
        # TODO: consider moving to self._remove_features_out
        for i in range(len(self.generators)):
            generator_group_valid = []
            for j in range(len(self.generators[i])):
                if self.generators[i][j].features_out:
                    generator_group_valid.append(self.generators[i][j])
            self.generators[i] = generator_group_valid

        return X, feature_metadata.type_group_map_special

    def _fit_transform_stage(
        self,
        X: DataFrame,
        generators: list["AbstractFeatureGenerator"],
        feature_metadata_in: FeatureMetadata,
        **kwargs,
    ) -> tuple[DataFrame, list["AbstractFeatureGenerator"], FeatureMetadata]:
        feature_df_list = []
        generator_group_valid = []
        for generator in generators:
            if generator.is_valid_metadata_in(feature_metadata_in):
                if generator.verbosity > self.verbosity:
                    generator.verbosity = self.verbosity
                generator.set_log_prefix(log_prefix=self.log_prefix + "\t\t", prepend=True)
                feature_df_list.append(generator.fit_transform(X, feature_metadata_in=feature_metadata_in, **kwargs))
                generator_group_valid.append(generator)
            else:
                self._log(15, f"\t\tSkipping {generator.__class__.__name__}: No input feature with required dtypes.")

        generators = generator_group_valid

        generators = [
            generator
            for j, generator in enumerate(generators)
            if feature_df_list[j] is not None and len(feature_df_list[j].columns) > 0
        ]
        feature_df_list = [
            feature_df for feature_df in feature_df_list if feature_df is not None and len(feature_df.columns) > 0
        ]

        if generators:
            # Raise an exception if generators expect different raw input types for the same feature.
            FeatureMetadata.join_metadatas(
                [generator.feature_metadata_in for generator in generators],
                shared_raw_features="error_if_diff",
            )

        feature_metadata = self._merge_feature_metadata(
            feature_metadata_lst=[generator.feature_metadata for generator in generators],
            shared_raw_features="error",
        )

        X = self._concat_features(
            feature_df_list=feature_df_list,
            index=X.index,
        )
        return X, generators, feature_metadata

    def _transform(self, X: DataFrame) -> DataFrame:
        for generator_group in self.generators:
            feature_df_list = []
            for generator in generator_group:
                feature_df_list.append(generator.transform(X))

            X = self._concat_features(
                feature_df_list=feature_df_list,
                index=X.index,
            )
        return X

    def _transform_stage(
        self,
        X: DataFrame,
        generators: list["AbstractFeatureGenerator"],
    ) -> DataFrame:
        feature_df_list = []
        for generator in generators:
            feature_df_list.append(generator.transform(X))

        X = self._concat_features(
            feature_df_list=feature_df_list,
            index=X.index,
        )
        return X

    def get_feature_links_chain(self):
        feature_links_chain = []
        for i in range(len(self.generators)):
            feature_links_group = {}
            for generator in self.generators[i]:
                feature_links = generator.get_feature_links()
                for feature_in, features_out in feature_links.items():
                    if feature_in in feature_links_group:
                        feature_links_group[feature_in] += features_out
                    else:
                        feature_links_group[feature_in] = features_out
            feature_links_chain.append(feature_links_group)
        return feature_links_chain

    def _remove_unused_features(self, feature_links_chain):
        unused_features_by_stage = self._get_unused_features(feature_links_chain)
        if unused_features_by_stage:
            unused_features_in = [
                feature
                for feature in self.feature_metadata_in.get_features()
                if feature in unused_features_by_stage[0]
            ]
            feature_metadata_in_unused = self.feature_metadata_in.keep_features(features=unused_features_in)
            if self._feature_metadata_in_unused:
                self._feature_metadata_in_unused = self._feature_metadata_in_unused.join_metadata(
                    feature_metadata_in_unused
                )
            else:
                self._feature_metadata_in_unused = feature_metadata_in_unused
            self._remove_features_in(features=unused_features_in)

        for i, generator_group in enumerate(self.generators):
            unused_features_in_stage = unused_features_by_stage[i]
            unused_features_out_stage = [
                feature_links_chain[i][feature_in]
                for feature_in in unused_features_in_stage
                if feature_in in feature_links_chain[i]
            ]
            unused_features_out_stage = list(
                set([feature for sublist in unused_features_out_stage for feature in sublist])
            )
            for generator in generator_group:
                unused_features_out_generator = [
                    feature for feature in generator.features_out if feature in unused_features_out_stage
                ]
                generator._remove_features_out(features=unused_features_out_generator)

    def _get_unused_features(self, feature_links_chain):
        features_in_list = []
        for i in range(len(self.generators)):
            stage = i + 1
            if stage > 1:
                if self.generators[stage - 2]:
                    features_in = FeatureMetadata.join_metadatas(
                        [generator.feature_metadata for generator in self.generators[stage - 2]],
                        shared_raw_features="error",
                    ).get_features()
                else:
                    features_in = []
            else:
                features_in = self.features_in
            features_in_list.append(features_in)
        return self._get_unused_features_generic(
            feature_links_chain=feature_links_chain, features_in_list=features_in_list
        )

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()
