import itertools
import pytest

from autogluon.core.features.feature_metadata import FeatureMetadata


def test_feature_metadata(data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    expected_feature_metadata_full = {
        ('category', ()): ['cat'],
        ('datetime', ()): ['datetime'],
        ('float', ()): ['float'],
        ('int', ()): ['int'],
        ('object', ()): ['obj'],
        ('object', ('datetime_as_object',)): ['datetime_as_object'],
        ('object', ('text',)): ['text']
    }

    expected_feature_metadata_get_features = ['int', 'float', 'obj', 'cat', 'datetime', 'text', 'datetime_as_object']

    expected_type_map_raw = {
        'cat': 'category',
        'datetime': 'datetime',
        'datetime_as_object': 'object',
        'float': 'float',
        'int': 'int',
        'obj': 'object',
        'text': 'object'
    }

    expected_type_group_map_special = {
        'datetime_as_object': ['datetime_as_object'],
        'text': ['text']
    }

    expected_feature_metadata_renamed_full = {
        ('category', ()): ['cat'],
        ('datetime', ()): ['datetime'],
        ('float', ()): ['obj'],
        ('int', ()): ['int_renamed'],
        ('object', ()): ['float'],
        ('object', ('datetime_as_object',)): ['datetime_as_object'],
        ('object', ('text',)): ['text_renamed']
    }

    expected_feature_metadata_recombined_full_full = {
        ('category', ()): ['cat'],
        ('custom_raw_type', ('custom_special_type',)): ['new_feature'],
        ('datetime', ()): ['datetime'],
        ('float', ()): ['float'],
        ('int', ('custom_special_type',)): ['int'],
        ('object', ()): ['obj'],
        ('object', ('datetime_as_object',)): ['datetime_as_object'],
        ('object', ('text',)): ['text']
    }

    # When
    feature_metadata = FeatureMetadata.from_df(input_data)
    feature_metadata_renamed = feature_metadata.rename_features(rename_map={'text': 'text_renamed', 'int': 'int_renamed', 'obj': 'float', 'float': 'obj'})
    feature_metadata_remove = feature_metadata.remove_features(features=['text', 'obj', 'float'])
    feature_metadata_keep = feature_metadata.keep_features(features=['text', 'obj', 'float'])
    feature_metadata_custom = FeatureMetadata(
        type_map_raw={'int': 'int', 'new_feature': 'custom_raw_type'},
        type_group_map_special={'custom_special_type': ['int', 'new_feature']}
    )
    feature_metadata_recombined = feature_metadata_keep.join_metadata(feature_metadata_remove)
    feature_metadata_recombined_alternate = FeatureMetadata.join_metadatas(metadata_list=[feature_metadata_keep, feature_metadata_remove])
    feature_metadata_recombined_full = FeatureMetadata.join_metadatas(metadata_list=[feature_metadata_keep, feature_metadata_remove, feature_metadata_custom], shared_raw_features='error_if_diff')

    # Therefore
    with pytest.raises(AssertionError):
        # Error because special contains feature not in raw
        FeatureMetadata(
            type_map_raw={'int': 'int'},
            type_group_map_special={'custom_special_type': ['int', 'new_feature']}
        )
    with pytest.raises(AssertionError):
        # Error because renaming to another existing feature without also renaming that feature
        feature_metadata.rename_features(rename_map={'text': 'obj'})
    with pytest.raises(KeyError):
        # Error if removing unknown feature
        feature_metadata_remove.remove_features(features=['text'])
    with pytest.raises(KeyError):
        # Error if getting unknown feature type
        feature_metadata_remove.get_feature_type_raw('text')
    with pytest.raises(KeyError):
        # Error if getting unknown feature type
        feature_metadata_remove.get_feature_types_special('text')
    with pytest.raises(AssertionError):
        # Error because feature_metadata_remove and feature_metadata_custom share a raw feature
        FeatureMetadata.join_metadatas(metadata_list=[feature_metadata_keep, feature_metadata_remove, feature_metadata_custom])

    assert feature_metadata.to_dict(inverse=True) == expected_feature_metadata_full
    assert feature_metadata.get_features() == expected_feature_metadata_get_features
    assert feature_metadata.type_map_raw == expected_type_map_raw
    assert dict(feature_metadata.type_group_map_special) == expected_type_group_map_special

    assert feature_metadata.get_feature_type_raw('text') == 'object'
    assert feature_metadata.get_feature_types_special('text') == ['text']
    assert feature_metadata.get_feature_type_raw('int') == 'int'
    assert feature_metadata.get_feature_types_special('int') == []
    assert feature_metadata_recombined_full.get_feature_types_special('int') == ['custom_special_type']
    assert feature_metadata_recombined_full.get_feature_type_raw('new_feature') == 'custom_raw_type'

    assert feature_metadata_renamed.to_dict(inverse=True) == expected_feature_metadata_renamed_full
    assert feature_metadata_recombined.to_dict() == feature_metadata.to_dict()
    assert feature_metadata_recombined_alternate.to_dict() == feature_metadata.to_dict()
    assert feature_metadata_recombined_full.to_dict(inverse=True) == expected_feature_metadata_recombined_full_full


def test_feature_metadata_get_features():
    type_map_raw = dict(
        a='1',
        b='2',
        c='3',
        d='1',
        e='1',
        f='4',
    )

    type_group_map_special = {
        's1': ['a', 'b', 'd'],
        's2': ['a', 'e'],
        's3': ['a', 'b'],
        's4': ['f']
    }

    expected_get_features = ['a', 'b', 'c', 'd', 'e', 'f']

    feature_metadata = FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)

    assert feature_metadata.get_features() == expected_get_features

    assert feature_metadata.get_features(valid_raw_types=['1']) == ['a', 'd', 'e']
    assert feature_metadata.get_features(valid_raw_types=['1', '3']) == ['a', 'c', 'd', 'e']
    assert feature_metadata.get_features(valid_raw_types=['UNKNOWN']) == []

    assert feature_metadata.get_features(valid_special_types=['s2', 's3']) == ['a', 'b', 'c', 'e']
    assert feature_metadata.get_features(valid_special_types=['s4']) == ['c', 'f']
    assert feature_metadata.get_features(valid_special_types=[]) == ['c']
    assert feature_metadata.get_features(valid_special_types=['UNKNOWN']) == ['c']

    assert feature_metadata.get_features(invalid_raw_types=[]) == expected_get_features
    assert feature_metadata.get_features(invalid_raw_types=['1', '3']) == ['b', 'f']
    assert feature_metadata.get_features(invalid_raw_types=['UNKNOWN']) == expected_get_features

    assert feature_metadata.get_features(invalid_special_types=['UNKNOWN']) == expected_get_features
    assert feature_metadata.get_features(invalid_special_types=[]) == expected_get_features
    assert feature_metadata.get_features(invalid_special_types=['s2', 's4']) == ['b', 'c', 'd']

    assert feature_metadata.get_features(required_special_types=['s2']) == ['a', 'e']
    assert feature_metadata.get_features(required_special_types=['s2', 's3']) == ['a']
    assert feature_metadata.get_features(required_special_types=['s2', 's4']) == []
    assert feature_metadata.get_features(required_special_types=['UNKNOWN']) == []

    assert feature_metadata.get_features(required_special_types=['s2'], required_exact=True) == ['e']
    assert feature_metadata.get_features(required_special_types=['s1', 's2', 's3'], required_exact=True) == ['a']

    assert feature_metadata.get_features(required_at_least_one_special=True) == ['a', 'b', 'd', 'e', 'f']

    assert feature_metadata.get_features(required_raw_special_pairs=[
        ('1', ['s2']),
    ]) == ['a', 'e']
    assert feature_metadata.get_features(required_raw_special_pairs=[
        ('1', None),
    ]) == ['a', 'd', 'e']
    assert feature_metadata.get_features(required_raw_special_pairs=[
        ('1', ['s2']),
        (None, ['s4']),
        ('3', None),
    ]) == ['a', 'c', 'e', 'f']
    assert feature_metadata.get_features(required_raw_special_pairs=[
        ('1', ['s2']),
        (None, ['s4']),
        ('3', None),
    ], required_exact=True) == ['c', 'e', 'f']

    # Assert that valid_raw_types is the opposite of invalid_raw_types through all combinations
    raw_types_to_check = ['1', '2', '3', '4', 'UNKNOWN']
    for L in range(0, len(raw_types_to_check) + 1):
        for subset in itertools.combinations(raw_types_to_check, L):
            valid_raw_types = list(subset)
            invalid_raw_types = [raw_type for raw_type in raw_types_to_check if raw_type not in valid_raw_types]
            assert feature_metadata.get_features(valid_raw_types=valid_raw_types) == feature_metadata.get_features(invalid_raw_types=invalid_raw_types)

    # Combined arguments
    assert feature_metadata.get_features(invalid_special_types=['s2', 's3'], required_special_types=['s1']) == ['d']
    assert feature_metadata.get_features(valid_raw_types=['2', '3'], valid_special_types=['s1']) == ['b', 'c']
    assert feature_metadata.get_features(valid_raw_types=['2', '3'], valid_special_types=['s1'], required_at_least_one_special=True) == ['b']
    assert feature_metadata.get_features(valid_raw_types=['2', '3'], required_special_types=['s1']) == ['b']
    assert feature_metadata.get_features(valid_raw_types=['2', '3'], required_special_types=['s1'], required_exact=True) == []
    assert feature_metadata.get_features(valid_raw_types=['2', '3'], required_special_types=['s1', 's3']) == ['b']
    assert feature_metadata.get_features(valid_raw_types=['2', '3'], required_special_types=['s1', 's3'], required_exact=True) == ['b']
