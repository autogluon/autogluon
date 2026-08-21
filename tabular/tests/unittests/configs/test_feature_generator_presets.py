import pytest

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.features.generators import AutoMLPipelineFeatureGenerator, IdentityFeatureGenerator
from autogluon.tabular.configs.feature_generator_presets import get_default_feature_generator


def test_get_default_feature_generator_none_returns_identity():
    feature_generator = get_default_feature_generator(None)

    assert isinstance(feature_generator, IdentityFeatureGenerator)


def test_get_default_feature_generator_auto_returns_automl_pipeline():
    feature_generator = get_default_feature_generator("auto")

    assert isinstance(feature_generator, AutoMLPipelineFeatureGenerator)


@pytest.mark.parametrize("invalid_feature_generator", [False, [], object()])
def test_get_default_feature_generator_invalid_type_raises_clear_value_error(invalid_feature_generator):
    feature_metadata = FeatureMetadata(type_map_raw={"feature": "int"})

    with pytest.raises(
        ValueError, match="To disable automated feature generation, pass None or IdentityFeatureGenerator"
    ):
        get_default_feature_generator(invalid_feature_generator, feature_metadata=feature_metadata)
