
from pandas import DataFrame

from autogluon.features.generators import DummyFeatureGenerator


def test_dummy_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    generator = DummyFeatureGenerator()

    expected_feature_metadata_in_full = {}
    expected_feature_metadata_full = {('int', ()): ['__dummy__']}

    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full
    )

    assert output_data.equals(DataFrame(data=[0, 0, 0, 0, 0, 0, 0, 0, 0], columns=['__dummy__']))
