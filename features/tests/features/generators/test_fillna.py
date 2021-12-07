import copy

import numpy as np

from autogluon.features.generators import FillNaFeatureGenerator


# TODO: Consider adding test of loading from csv
def test_fillna_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()
    expected_output_data = input_data.fillna(
        {
            'int': np.nan,
            'float': np.nan,
            'obj': '',
            'cat': np.nan,
            'datetime': np.nan,
            'text': '',
            'datetime_as_object': ''
        },
        downcast=False
    )

    generator = FillNaFeatureGenerator()

    expected_feature_metadata_in_full = {
        ('category', ()): ['cat'],
        ('datetime', ()): ['datetime'],
        ('float', ()): ['float'],
        ('int', ()): ['int'],
        ('object', ()): ['obj'],
        ('object', ('datetime_as_object',)): ['datetime_as_object'],
        ('object', ('text',)): ['text']
    }

    expected_feature_metadata_full = expected_feature_metadata_in_full

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    assert expected_output_data.equals(output_data)


# Edge case when input is all integers but is of object type without NaN values. Without downcast=False, it will be converted to int type.
# This test confirms that this unwanted conversion does not happen.
def test_fillna_object_edgecase_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_int_feature().to_frame('int_as_object')
    input_data = input_data.astype('object')
    edge_case_input = copy.deepcopy(input_data)
    input_data.loc[0] = np.nan
    expected_output_data = copy.deepcopy(input_data)
    expected_output_data.loc[0] = ''

    generator = FillNaFeatureGenerator()

    expected_feature_metadata_in_full = {
        ('object', ()): ['int_as_object'],
    }

    expected_feature_metadata_full = expected_feature_metadata_in_full

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    edge_case_output = generator.transform(edge_case_input)

    assert expected_output_data.equals(output_data)
    assert edge_case_input.equals(edge_case_output)