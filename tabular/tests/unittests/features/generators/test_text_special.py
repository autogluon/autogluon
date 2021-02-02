
from autogluon.features.generators import TextSpecialFeatureGenerator


def test_text_special_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    generator = TextSpecialFeatureGenerator()

    expected_feature_metadata_in_full = {
        ('object', ('text',)): ['text'],
    }
    expected_feature_metadata_full = {('int', ('binned', 'text_special')): [
        'text.char_count',
        'text.word_count',
        'text.capital_ratio',
        'text.lower_ratio',
        'text.special_ratio',
        'text.symbol_ratio. '
    ]}

    expected_output_data_feat_lower_ratio = [3, 2, 0, 3, 3, 3, 3, 3, 1]

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    assert expected_output_data_feat_lower_ratio == list(output_data['text.lower_ratio'].values)
