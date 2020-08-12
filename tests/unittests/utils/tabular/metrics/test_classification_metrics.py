import numpy as np
import pytest
from autogluon.utils.tabular.metrics.classification_metrics import confusion_matrix


def test_confusion_matrix_with_valid_inputs_without_labels_and_weights():
    # Given
    input_solution = [2, 0, 2, 2, 0, 1]
    input_prediction = [0, 0, 2, 2, 0, 2]
    expected_output = np.array([[2, 0, 0], [0, 0, 1], [1, 0, 2]])

    # When
    observed_output = confusion_matrix(input_solution, input_prediction)

    # Then
    assert(np.array_equal(expected_output, observed_output))


def test_confusion_matrix_with_valid_inputs_with_labels_and_without_weights():
    # Given
    input_solution = ["cat", "ant", "cat", "cat", "ant", "bird"]
    input_prediction = ["ant", "ant", "cat", "cat", "ant", "cat"]
    labels = ["ant", "bird", "cat"]
    expected_output = np.array([[2, 0, 0], [0, 0, 1], [1, 0, 2]])

    # When
    observed_output = confusion_matrix(input_solution, input_prediction, labels=labels)

    # Then
    assert(np.array_equal(expected_output, observed_output))


def test_confusion_matrix_with_valid_inputs_with_labels_and_with_weights():
    # Given
    input_solution = ["cat", "ant", "cat", "cat", "ant", "bird"]
    input_prediction = ["ant", "ant", "cat", "cat", "ant", "cat"]
    labels = ["ant", "bird", "cat"]
    weights = [0.1, 0.3, 1.0, 0.8, 0.2, 2.0]
    expected_output = np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 2.0], [0.1, 0.0, 1.8]])

    # When
    observed_output = confusion_matrix(input_solution, input_prediction, labels=labels, weights=weights)

    # Then
    assert(np.array_equal(expected_output, observed_output))


def test_confusion_matrix_with_valid_inputs_with_lesser_number_of_labels_and_without_weights():
    # Given
    input_solution = ["cat", "ant", "cat", "cat", "ant", "bird"]
    input_prediction = ["ant", "ant", "cat", "cat", "ant", "cat"]
    labels = ["bird", "cat"]
    expected_output = np.array([[0, 1], [0, 2]])

    # When
    observed_output = confusion_matrix(input_solution, input_prediction, labels=labels)

    # Then
    assert(np.array_equal(expected_output, observed_output))


def test_confusion_matrix_with_unequal_samples():
    # Given
    input_solution = ["cat", "ant", "cat"]
    input_prediction = ["ant", "ant", "cat", "cat", "ant", "cat"]

    # When-Then
    with pytest.raises(ValueError):
        observed_output = confusion_matrix(input_solution, input_prediction)


def test_confusion_matrix_with_multioutput_samples():
    # Given
    input_solution = [["cat", "ant", "cat"]]
    input_prediction = [["ant", "ant", "cat"]]

    # When-Then
    with pytest.raises(ValueError):
        observed_output = confusion_matrix(input_solution, input_prediction)


def test_confusion_matrix_with_empty_labels():
    # Given
    input_solution = ["cat", "ant", "cat", "cat", "ant", "bird"]
    input_prediction = ["ant", "ant", "cat", "cat", "ant", "cat"]
    labels = []

    # When-Then
    with pytest.raises(ValueError):
        observed_output = confusion_matrix(input_solution, input_prediction, labels=labels)


def test_confusion_matrix_with_multiDimensional_labels():
    # Given
    input_solution = ["cat", "ant", "cat", "cat", "ant", "bird"]
    input_prediction = ["ant", "ant", "cat", "cat", "ant", "cat"]
    labels = [["ant", "bird"], "cat"]

    # When-Then
    with pytest.raises(ValueError):
        observed_output = confusion_matrix(input_solution, input_prediction, labels=labels)


def test_confusion_matrix_with_invalid_weights():
    # Given
    input_solution = ["cat", "ant", "cat", "cat", "ant", "bird"]
    input_prediction = ["ant", "ant", "cat", "cat", "ant", "cat"]
    labels = [[1, 2], 0.1, [0.1], 3, 1]

    # When-Then
    with pytest.raises(ValueError):
        observed_output = confusion_matrix(input_solution, input_prediction, labels=labels)

def test_confusion_matrix_with_empty_inputs():
    # Given
    input_solution = []
    input_prediction = []
    labels = ["bird", "cat"]
    expected_output = np.array([[0, 0], [0, 0]])

    # When
    observed_output = confusion_matrix(input_solution, input_prediction, labels = labels)

    # Then
    assert(np.array_equal(expected_output, observed_output))
