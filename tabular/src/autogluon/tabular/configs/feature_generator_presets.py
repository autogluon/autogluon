import copy

from autogluon.features.generators import (
    AbstractFeatureGenerator,
    AutoMLInterpretablePipelineFeatureGenerator,
    AutoMLPipelineFeatureGenerator,
    IdentityFeatureGenerator,
)


def get_default_feature_generator(feature_generator, feature_metadata=None, init_kwargs=None):
    if init_kwargs is None:
        init_kwargs = dict()
    if feature_generator is None:
        feature_generator = IdentityFeatureGenerator()
    elif isinstance(feature_generator, str):
        if feature_generator == "auto":
            feature_generator = AutoMLPipelineFeatureGenerator(**init_kwargs)
        elif feature_generator == "interpretable":
            feature_generator = AutoMLInterpretablePipelineFeatureGenerator(**init_kwargs)
        else:
            raise ValueError(
                f"Unknown feature_generator preset: '{feature_generator}', valid presets: {['auto', 'interpretable']}"
            )
    elif not isinstance(feature_generator, AbstractFeatureGenerator):
        raise ValueError(
            "`feature_generator` must be None, one of the preset strings "
            f"{['auto', 'interpretable']}, or an `AbstractFeatureGenerator` instance. "
            "To disable automated feature generation, pass None or IdentityFeatureGenerator(). "
            f"Received {type(feature_generator).__name__}: {feature_generator!r}"
        )
    if feature_metadata is not None:
        if feature_generator.feature_metadata_in is None and not feature_generator.is_fit():
            feature_generator.feature_metadata_in = copy.deepcopy(feature_metadata)
        else:
            raise AssertionError("`feature_metadata_in` already exists in `feature_generator`.")
    return feature_generator
