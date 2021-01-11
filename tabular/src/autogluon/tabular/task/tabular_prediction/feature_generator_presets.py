
import copy

from ...features import AutoMLPipelineFeatureGenerator, IdentityFeatureGenerator


def get_default_feature_generator(feature_generator, feature_metadata=None, init_kwargs=None):
    if init_kwargs is None:
        init_kwargs = dict()
    if feature_generator is None:
        feature_generator = IdentityFeatureGenerator()
    elif isinstance(feature_generator, str):
        if feature_generator == 'auto':
            feature_generator = AutoMLPipelineFeatureGenerator(**init_kwargs)
        else:
            raise ValueError(f"Unknown feature_generator preset: '{feature_generator}', valid presets: {['auto']}")
    if feature_metadata is not None:
        if feature_generator.feature_metadata_in is None and not feature_generator.is_fit():
            feature_generator.feature_metadata_in = copy.deepcopy(feature_metadata)
        else:
            raise AssertionError('`feature_metadata_in` already exists in `feature_generator`.')
    return feature_generator
