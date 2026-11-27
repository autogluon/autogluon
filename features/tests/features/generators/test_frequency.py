import numpy as np
import pandas as pd
import pytest

from autogluon.features.generators import FrequencyFeatureGenerator


@pytest.mark.parametrize("dtype", ["object", "category"])
@pytest.mark.parametrize("keep_original", [True, False])
@pytest.mark.parametrize("log", [True, False])
def test_frequency_fit_transform_matches_transform(generator_helper, dtype, keep_original, log):
    X = pd.DataFrame({"category": ["a", "a", "a", "b"], "number": [1, 2, 3, 4]}, index=[10, 20, 30, 40])
    X["category"] = X["category"].astype(dtype)
    generator = FrequencyFeatureGenerator(keep_original=keep_original, log=log)

    output = generator_helper.fit_transform_assert(input_data=X, generator=generator)

    encoded_name = "category_freq_log" if log else "category_freq"
    expected = X.copy() if keep_original else X[["number"]].copy()
    frequencies = np.array([3.0, 3.0, 3.0, 1.0])
    expected[encoded_name] = np.log1p(frequencies) if log else frequencies
    pd.testing.assert_frame_equal(output, expected)
    assert output.columns.is_unique
    assert generator.features_out == list(expected.columns)

    held_out = pd.DataFrame({"category": ["a", "unseen", None], "number": [5, 6, 7]}, index=[50, 60, 70])
    held_out["category"] = held_out["category"].astype(dtype)
    expected_held_out = held_out.copy() if keep_original else held_out[["number"]].copy()
    frequencies = np.array([3.0, 0.0, 0.0])
    expected_held_out[encoded_name] = np.log1p(frequencies) if log else frequencies
    pd.testing.assert_frame_equal(generator.transform(held_out), expected_held_out)
