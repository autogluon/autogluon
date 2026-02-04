import numpy as np
import pandas as pd
import pytest

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS
from autogluon.core.data.label_cleaner import (
    LabelCleaner,
    LabelCleanerBinary,
    LabelCleanerDummy,
    LabelCleanerMulticlass,
    LabelCleanerMulticlassToBinary,
)


def test_label_cleaner_binary():
    # Given
    problem_type = BINARY
    input_labels_numpy = np.array(["l1", "l2", "l2", "l1", "l1", "l2"])
    input_labels = pd.Series(input_labels_numpy)
    input_labels_category = input_labels.astype("category")
    input_labels_with_shifted_index = input_labels.copy()
    input_labels_with_shifted_index.index += 5
    input_labels_new = np.array(["new", "l1", "l2"])
    expected_output_labels = pd.Series([0, 1, 1, 0, 0, 1], dtype="uint8")
    expected_output_labels_pos_class_l1 = pd.Series([1, 0, 0, 1, 1, 0], dtype="uint8")
    expected_output_labels_new = pd.Series([np.nan, 0, 1])
    expected_output_labels_new_pos_class_l1 = pd.Series([np.nan, 1, 0])
    expected_output_labels_new_inverse = pd.Series([np.nan, "l1", "l2"])

    # When
    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=input_labels)  # positive_class='l2'
    label_cleaner_pos_class_l1 = LabelCleaner.construct(problem_type=problem_type, y=input_labels, positive_class="l1")

    # Raise exception
    with pytest.raises(ValueError):
        LabelCleaner.construct(problem_type=problem_type, y=input_labels, positive_class="unknown_class")

    # Raise exception
    with pytest.raises(AssertionError):
        LabelCleaner.construct(problem_type=problem_type, y=input_labels_new)

    # Then
    assert isinstance(label_cleaner, LabelCleanerBinary)
    assert label_cleaner.problem_type_transform == BINARY
    assert label_cleaner.cat_mappings_dependent_var == {0: "l1", 1: "l2"}
    assert label_cleaner_pos_class_l1.cat_mappings_dependent_var == {0: "l2", 1: "l1"}

    output_labels = label_cleaner.transform(input_labels)
    output_labels_pos_class_l1 = label_cleaner_pos_class_l1.transform(input_labels)
    output_labels_with_numpy = label_cleaner.transform(input_labels_numpy)
    output_labels_category = label_cleaner.transform(input_labels_category)
    output_labels_with_shifted_index = label_cleaner.transform(input_labels_with_shifted_index)
    output_labels_new = label_cleaner.transform(input_labels_new)
    output_labels_new_pos_class_l1 = label_cleaner_pos_class_l1.transform(input_labels_new)

    output_labels_inverse = label_cleaner.inverse_transform(output_labels)
    output_labels_inverse_pos_class_l1 = label_cleaner_pos_class_l1.inverse_transform(output_labels_pos_class_l1)
    output_labels_with_shifted_index_inverse = label_cleaner.inverse_transform(output_labels_with_shifted_index)
    output_labels_new_inverse = label_cleaner.inverse_transform(output_labels_new)
    output_labels_new_inverse_pos_class_l1 = label_cleaner_pos_class_l1.inverse_transform(
        output_labels_new_pos_class_l1
    )

    assert expected_output_labels.equals(output_labels)
    assert expected_output_labels_pos_class_l1.equals(output_labels_pos_class_l1)
    assert expected_output_labels.equals(output_labels_with_numpy)
    assert expected_output_labels.equals(output_labels_category)
    assert not expected_output_labels.equals(output_labels_with_shifted_index)
    output_labels_with_shifted_index.index -= 5
    assert expected_output_labels.equals(output_labels_with_shifted_index)
    assert expected_output_labels_new.equals(output_labels_new)
    assert expected_output_labels_new_pos_class_l1.equals(output_labels_new_pos_class_l1)

    assert input_labels.equals(output_labels_inverse)
    assert input_labels.equals(output_labels_inverse_pos_class_l1)
    assert input_labels_with_shifted_index.equals(output_labels_with_shifted_index_inverse)
    assert expected_output_labels_new_inverse.equals(output_labels_new_inverse)
    assert expected_output_labels_new_inverse.equals(output_labels_new_inverse_pos_class_l1)


def test_label_cleaner_multiclass():
    # Given
    problem_type = MULTICLASS
    input_labels_numpy = np.array([2, 4, 2, 2, 4, 1], dtype="int32")
    input_labels = pd.Series(input_labels_numpy, dtype="int32")
    input_labels_category = input_labels.astype("category")
    input_labels_with_shifted_index = input_labels.copy()
    input_labels_with_shifted_index.index += 5
    input_labels_new = np.array([3, 5, 2], dtype="int32")
    expected_output_labels = pd.Series([1, 2, 1, 1, 2, 0], dtype="uint8")
    expected_output_labels_new = pd.Series([np.nan, np.nan, 1])
    expected_output_labels_new_inverse = pd.Series([np.nan, np.nan, 2])

    # When
    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=input_labels, y_uncleaned=input_labels)

    # Then
    assert isinstance(label_cleaner, LabelCleanerMulticlass)
    assert label_cleaner.problem_type_transform == MULTICLASS
    assert label_cleaner.cat_mappings_dependent_var == {0: 1, 1: 2, 2: 4}

    output_labels = label_cleaner.transform(input_labels)
    output_labels_with_numpy = label_cleaner.transform(input_labels_numpy)
    output_labels_category = label_cleaner.transform(input_labels_category)
    output_labels_with_shifted_index = label_cleaner.transform(input_labels_with_shifted_index)
    output_labels_new = label_cleaner.transform(input_labels_new)

    output_labels_inverse = label_cleaner.inverse_transform(output_labels)
    output_labels_with_shifted_index_inverse = label_cleaner.inverse_transform(output_labels_with_shifted_index)
    output_labels_new_inverse = label_cleaner.inverse_transform(output_labels_new)

    output_labels_uncleaned = label_cleaner.transform_pred_uncleaned(y=input_labels)
    output_labels_uncleaned_inverse = label_cleaner.inverse_transform_pred_uncleaned(y=output_labels_uncleaned)

    output_labels_uncleaned_new = label_cleaner.transform_pred_uncleaned(y=input_labels_new)
    output_labels_uncleaned_new_inverse = label_cleaner.inverse_transform_pred_uncleaned(y=output_labels_uncleaned_new)

    input_labels = input_labels.astype("int32")
    output_labels_uncleaned_inverse = output_labels_uncleaned_inverse.astype("int32")

    assert expected_output_labels.equals(output_labels)
    assert expected_output_labels.equals(output_labels_with_numpy)
    assert expected_output_labels.equals(output_labels_category)
    assert not expected_output_labels.equals(output_labels_with_shifted_index)
    output_labels_with_shifted_index.index -= 5
    assert expected_output_labels.equals(output_labels_with_shifted_index)
    assert expected_output_labels_new.equals(output_labels_new)

    assert input_labels.equals(output_labels_inverse)
    assert input_labels_with_shifted_index.equals(output_labels_with_shifted_index_inverse)
    assert expected_output_labels_new_inverse.equals(output_labels_new_inverse)
    assert input_labels.equals(output_labels_uncleaned_inverse)
    assert expected_output_labels_new_inverse.equals(output_labels_uncleaned_new_inverse)


def test_label_cleaner_multiclass_to_binary():
    # Given
    problem_type = MULTICLASS
    input_labels_numpy = np.array(["l1", "l2", "l2", "l1", "l1", "l2"])
    input_labels = pd.Series(input_labels_numpy)
    input_labels_uncleaned = pd.Series(["l0", "l1", "l2", "l2", "l1", "l1", "l2", "l3", "l4"])
    input_labels_category = input_labels.astype("category")
    input_labels_with_shifted_index = input_labels.copy()
    input_labels_with_shifted_index.index += 5
    input_labels_new = pd.Series(["l0", "l1", "l2"])
    input_labels_new_with_unknown = pd.Series(["l0", "l1", "l2", "UNKNOWN_1", "l4", "UNKNOWN_2"])
    input_labels_proba_transformed = pd.Series([0.7, 0.2, 0.5], index=[5, 2, 8])
    expected_output_labels = pd.Series([0, 1, 1, 0, 0, 1], dtype="uint8")
    expected_output_labels_new = pd.Series([np.nan, 0, 1])
    expected_output_labels_new_inverse = pd.Series([np.nan, "l1", "l2"])
    expected_output_labels_proba_transformed_inverse = pd.DataFrame(
        data=[[0, 0.3, 0.7, 0, 0], [0, 0.8, 0.2, 0, 0], [0, 0.5, 0.5, 0, 0]],
        index=[5, 2, 8],
        columns=["l0", "l1", "l2", "l3", "l4"],
        dtype=np.float32,
    )
    expected_output_labels_new_with_unknown = pd.Series([0, 1, 2, np.nan, 4, np.nan])
    expected_output_labels_new_with_unknown_inverse = pd.Series(["l0", "l1", "l2", np.nan, "l4", np.nan])

    # When
    label_cleaner = LabelCleaner.construct(
        problem_type=problem_type, y=input_labels, y_uncleaned=input_labels_uncleaned
    )

    # Then
    assert isinstance(label_cleaner, LabelCleanerMulticlassToBinary)
    assert label_cleaner.problem_type_transform == BINARY
    assert label_cleaner.cat_mappings_dependent_var == {0: "l1", 1: "l2"}

    output_labels = label_cleaner.transform(input_labels)
    output_labels_with_numpy = label_cleaner.transform(input_labels_numpy)
    output_labels_category = label_cleaner.transform(input_labels_category)
    output_labels_with_shifted_index = label_cleaner.transform(input_labels_with_shifted_index)
    output_labels_new = label_cleaner.transform(input_labels_new)

    output_labels_inverse = label_cleaner.inverse_transform(output_labels)
    output_labels_with_shifted_index_inverse = label_cleaner.inverse_transform(output_labels_with_shifted_index)
    output_labels_new_inverse = label_cleaner.inverse_transform(output_labels_new)

    output_labels_uncleaned = label_cleaner.transform_pred_uncleaned(y=input_labels)
    output_labels_uncleaned_inverse = label_cleaner.inverse_transform_pred_uncleaned(y=output_labels_uncleaned)

    output_labels_uncleaned_new = label_cleaner.transform_pred_uncleaned(y=input_labels_new)
    output_labels_uncleaned_new_inverse = label_cleaner.inverse_transform_pred_uncleaned(y=output_labels_uncleaned_new)

    output_labels_uncleaned_new_with_unknown = label_cleaner.transform_pred_uncleaned(y=input_labels_new_with_unknown)
    output_labels_uncleaned_new_with_unknown_inverse = label_cleaner.inverse_transform_pred_uncleaned(
        y=output_labels_uncleaned_new_with_unknown
    )

    assert expected_output_labels.equals(output_labels)
    assert expected_output_labels.equals(output_labels_with_numpy)
    assert expected_output_labels.equals(output_labels_category)
    assert not expected_output_labels.equals(output_labels_with_shifted_index)
    output_labels_with_shifted_index.index -= 5
    assert expected_output_labels.equals(output_labels_with_shifted_index)
    assert expected_output_labels_new.equals(output_labels_new)

    assert input_labels.equals(output_labels_inverse)
    assert input_labels_with_shifted_index.equals(output_labels_with_shifted_index_inverse)
    assert expected_output_labels_new_inverse.equals(output_labels_new_inverse)

    assert input_labels.equals(output_labels_uncleaned_inverse)
    assert input_labels_new.equals(output_labels_uncleaned_new_inverse)

    assert expected_output_labels_new_with_unknown.equals(output_labels_uncleaned_new_with_unknown)
    assert expected_output_labels_new_with_unknown_inverse.equals(output_labels_uncleaned_new_with_unknown_inverse)

    output_labels_proba_transformed_inverse = label_cleaner.inverse_transform_proba(
        input_labels_proba_transformed, as_pandas=True
    )

    pd.testing.assert_frame_equal(
        expected_output_labels_proba_transformed_inverse, output_labels_proba_transformed_inverse
    )


def test_label_cleaner_regression():
    # Given
    problem_type = REGRESSION
    input_labels_numpy = np.array([2, 4, 2, 2, 4, 1])
    input_labels = pd.Series(input_labels_numpy)
    input_labels_new = pd.Series([3, 5, 2])
    expected_output_labels = input_labels.copy()
    expected_output_labels_new = input_labels_new.copy()
    expected_output_labels_new_inverse = input_labels_new.copy()

    # When
    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=input_labels, y_uncleaned=None)

    # Then
    assert isinstance(label_cleaner, LabelCleanerDummy)
    assert label_cleaner.problem_type_transform == REGRESSION

    output_labels = label_cleaner.transform(input_labels)
    output_labels_with_numpy = label_cleaner.transform(input_labels_numpy)
    output_labels_new = label_cleaner.transform(input_labels_new)

    output_labels_inverse = label_cleaner.inverse_transform(output_labels)
    output_labels_new_inverse = label_cleaner.inverse_transform(output_labels_new)

    assert expected_output_labels.equals(output_labels)
    assert expected_output_labels.equals(output_labels_with_numpy)
    assert expected_output_labels_new.equals(output_labels_new)

    assert input_labels.equals(output_labels_inverse)
    assert expected_output_labels_new_inverse.equals(output_labels_new_inverse)


def test_label_softclass():
    # Given
    problem_type = SOFTCLASS
    input_labels = pd.DataFrame(
        [
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0.3, 0.6, 0.1, 0],
        ]
    )

    # When
    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=input_labels, y_uncleaned=None)

    # Then
    assert input_labels.equals(label_cleaner.transform(input_labels))
    assert input_labels.equals(label_cleaner.inverse_transform(input_labels))
    assert label_cleaner.num_classes == 6
