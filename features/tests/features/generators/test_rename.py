
from autogluon.features.generators import RenameFeatureGenerator


def test_rename(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    generator = RenameFeatureGenerator(name_prefix='pre_', name_suffix='_suf')

    expected_feature_metadata_in_full = {
        ('category', ()): ['cat'],
        ('datetime', ()): ['datetime'],
        ('float', ()): ['float'],
        ('int', ()): ['int'],
        ('object', ()): ['obj'],
        ('object', ('datetime_as_object',)): ['datetime_as_object'],
        ('object', ('text',)): ['text']
    }

    expected_feature_metadata_full = {
        ('category', ()): ['pre_cat_suf'],
        ('datetime', ()): ['pre_datetime_suf'],
        ('float', ()): ['pre_float_suf'],
        ('int', ()): ['pre_int_suf'],
        ('object', ()): ['pre_obj_suf'],
        ('object', ('datetime_as_object',)): ['pre_datetime_as_object_suf'],
        ('object', ('text',)): ['pre_text_suf']
    }

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    # Therefore
    output_data.columns = input_data.columns

    assert input_data.equals(output_data)
