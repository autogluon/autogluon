import numpy as np
import pandas as pd

from autogluon.common import FeatureMetadata
from autogluon.features.generators import DropDuplicatesFeatureGenerator


def test_drop_duplicates_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    generator = DropDuplicatesFeatureGenerator()

    og_feature_metadata_in = FeatureMetadata.from_df(input_data).to_dict(inverse=True)

    expected_og_feature_metadata_in = {
        ("category", ()): ["cat"],
        ("datetime", ()): ["datetime"],
        ("float", ()): ["float"],
        ("int", ()): ["int_bool", "int"],
        ("object", ()): ["obj"],
        ("object", ("datetime_as_object",)): ["datetime_as_object"],
        ("object", ("text",)): ["text"],
    }

    assert expected_og_feature_metadata_in == og_feature_metadata_in

    expected_feature_metadata_in_full = {
        ("category", ()): ["cat"],
        ("datetime", ()): ["datetime"],
        ("float", ()): ["float"],
        ("int", ()): ["int_bool", "int"],
        ("object", ()): ["obj"],
        ("object", ("datetime_as_object",)): ["datetime_as_object"],
        ("object", ("text",)): ["text"],
    }

    expected_feature_metadata_full = {
        ("category", ()): ["cat"],
        ("datetime", ()): ["datetime"],
        ("float", ()): ["float"],
        ("int", ()): ["int_bool", "int"],
        ("object", ()): ["obj"],
        ("object", ("datetime_as_object",)): ["datetime_as_object"],
        ("object", ("text",)): ["text"],
    }

    # When
    generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )


def test_drop_duplicates_feature_generator_with_dupes(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_duplicate()

    generator = DropDuplicatesFeatureGenerator()

    og_feature_metadata_in = FeatureMetadata.from_df(input_data).to_dict(inverse=True)

    expected_og_feature_metadata_in = {
        ("category", ()): ["cat"],
        ("datetime", ()): ["datetime"],
        ("float", ()): ["float"],
        ("int", ()): [
            "int_bool",
            "int",
            "int_bool_dup_1",
            "int_bool_dup_2",
            "int_bool_dup_3",
            "int_bool_dup_4",
            "int_bool_dup_5",
            "int_bool_dup_6",
            "int_bool_dup_7",
            "int_bool_dup_8",
            "int_bool_dup_9",
            "int_bool_dup_10",
            "int_bool_dup_11",
            "int_bool_dup_12",
            "int_bool_dup_13",
            "int_bool_dup_14",
            "int_bool_dup_15",
            "int_bool_dup_16",
            "int_bool_dup_17",
            "int_bool_dup_18",
            "int_bool_dup_19",
            "int_bool_dup_20",
            "int_bool_dup_final",
        ],
        ("object", ()): ["cat_useless", "obj"],
        ("object", ("datetime_as_object",)): ["datetime_as_object"],
        ("object", ("text",)): ["text"],
    }

    assert expected_og_feature_metadata_in == og_feature_metadata_in

    expected_feature_metadata_in_full = {
        ("category", ()): ["cat"],
        ("datetime", ()): ["datetime"],
        ("float", ()): ["float"],
        ("int", ()): ["int_bool", "int"],
        ("object", ()): ["cat_useless", "obj"],
        ("object", ("datetime_as_object",)): ["datetime_as_object"],
        ("object", ("text",)): ["text"],
    }

    expected_feature_metadata_full = {
        ("category", ()): ["cat"],
        ("datetime", ()): ["datetime"],
        ("float", ()): ["float"],
        ("int", ()): ["int_bool", "int"],
        ("object", ()): ["cat_useless", "obj"],
        ("object", ("datetime_as_object",)): ["datetime_as_object"],
        ("object", ("text",)): ["text"],
    }

    # When
    generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )


def test_drop_duplicates_numeric_edge_cases(generator_helper):
    df = pd.DataFrame(
        {
            "A": [0, 1, 2, 3, 4],
            "B": ["0", "1", "2", "3", "4"],
            "C": [np.nan, 1, 2, 3, 4],
            "D": [np.nan, 1, 2, 3, 4],
            "E": [4, 3, 2, 1, 4],
            "F": [4, 3, 2, 1, 0],
            "G": [4, 3, 2, 1, 4],
            "H": [4.0, 3.0, 2.0, 1.0, 4.0],
            "I": [4.2, 3.6, -4.3, 1.7, 4.2],
            "J": [4.2, 3.6, -4.3, 1.7, 4.2],
        }
    )

    feature_generator = DropDuplicatesFeatureGenerator()
    feature_metadata_in = FeatureMetadata.from_df(df)

    # Drop D because C and D are identical and both numerical, and C comes earlier
    # Drop G and H because E and G are identical and both numerical, and G comes earlier
    # Drop J because I and J are identical and both numerical, and I comes earlier
    expected_dropped_1 = ["D", "G", "H", "J"]
    actual_dropped_1 = feature_generator._drop_duplicate_features(X=df, feature_metadata_in=feature_metadata_in)
    assert expected_dropped_1 == actual_dropped_1

    og_feature_metadata_in = FeatureMetadata.from_df(df).to_dict(inverse=True)

    expected_og_feature_metadata_in = {
        ("float", ()): ["C", "D", "H", "I", "J"],
        ("int", ()): ["A", "E", "F", "G"],
        ("object", ()): ["B"],
    }

    assert expected_og_feature_metadata_in == og_feature_metadata_in

    expected_feature_metadata_in_full = {
        ("float", ()): ["C", "I"],
        ("int", ()): ["A", "E", "F"],
        ("object", ()): ["B"],
    }

    expected_feature_metadata_full = {("float", ()): ["C", "I"], ("int", ()): ["A", "E", "F"], ("object", ()): ["B"]}

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=df,
        generator=feature_generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    for removed_feature in expected_dropped_1:
        assert removed_feature not in output_data.columns


def test_drop_duplicates_category_edge_cases():
    df = pd.DataFrame(
        {
            "A": [0, 1, 2, 3, 4],
            "B": [0, 3, "b", 4, np.nan],
            "C": ["a", "b", 3, "d", "e"],
            "D": [0, 3, "b", 4, np.nan],
            "E": [5, np.nan, "b", 4, 3],
            "F": [4, 3, 2, 1, 0],
            "G": [4, 3, 2, 1, 4],
        }
    )
    feature_generator = DropDuplicatesFeatureGenerator()
    feature_metadata_in = FeatureMetadata.from_df(df)

    # Drop D because B and D are identical with same dtype, and B comes earlier
    expected_dropped_1 = ["D"]
    actual_dropped_1 = feature_generator._drop_duplicate_features(X=df, feature_metadata_in=feature_metadata_in)
    assert expected_dropped_1 == actual_dropped_1

    df2 = pd.DataFrame(
        {
            "A": [0, 1, 2, 3, 4],
            "D": [0, 3, "b", 4, np.nan],
            "C": ["a", "b", 3, "d", "e"],
            "B": [0, 3, "b", 4, np.nan],
            "E": [5, np.nan, "b", 4, 3],
            "F": [4, 3, 2, 1, 0],
            "G": [4, 3, 2, 1, 4],
        }
    )
    feature_metadata_in2 = FeatureMetadata.from_df(df2)
    # Drop B because B and D are identical with same dtype, and D comes earlier
    expected_dropped_2 = ["B"]
    actual_dropped_2 = feature_generator._drop_duplicate_features(X=df2, feature_metadata_in=feature_metadata_in2)
    assert expected_dropped_2 == actual_dropped_2

    df["D"] = df["D"].astype("category")
    feature_metadata_in = FeatureMetadata.from_df(df)

    # Drop nothing because even though B and D have same values, they aren't the same dtype and thus aren't safe to drop.
    expected_dropped_3 = []
    actual_dropped_3 = feature_generator._drop_duplicate_features(X=df, feature_metadata_in=feature_metadata_in)
    assert expected_dropped_3 == actual_dropped_3

    df["B"] = df["B"].astype("category")
    feature_metadata_in = FeatureMetadata.from_df(df)

    # Drop D because B and D are identical with same dtype, and B comes earlier
    expected_dropped_4 = ["D"]
    actual_dropped_4 = feature_generator._drop_duplicate_features(X=df, feature_metadata_in=feature_metadata_in)
    assert expected_dropped_4 == actual_dropped_4

    df["E"] = df["E"].astype("category")
    feature_metadata_in = FeatureMetadata.from_df(df)

    # Drop D and E because B and D are identical with same dtype, and B comes earlier, and E is functionally identical and comes later
    expected_dropped_5 = ["D", "E"]
    actual_dropped_5 = feature_generator._drop_duplicate_features(X=df, feature_metadata_in=feature_metadata_in)
    assert expected_dropped_5 == actual_dropped_5

    # Make everything category type
    df["A"] = df["A"].astype("category")
    df["C"] = df["C"].astype("category")
    df["F"] = df["F"].astype("category")
    df["G"] = df["G"].astype("category")
    feature_metadata_in = FeatureMetadata.from_df(df)

    # Drop all except A and G, because all others are equal to A functionally
    expected_dropped_6 = ["B", "C", "D", "E", "F"]
    actual_dropped_6 = feature_generator._drop_duplicate_features(X=df, feature_metadata_in=feature_metadata_in)
    assert expected_dropped_6 == actual_dropped_6

    # Make everything object type
    df = df.astype("object")
    feature_metadata_in = FeatureMetadata.from_df(df)

    # Drop only D, because B and D are identical with same dtype, and B is earlier
    expected_dropped_7 = ["D"]
    actual_dropped_7 = feature_generator._drop_duplicate_features(X=df, feature_metadata_in=feature_metadata_in)
    assert expected_dropped_7 == actual_dropped_7
