import numpy as np
import pandas as pd
import pytest

from autogluon.features.generators import CategoricalInteractionFeatureGenerator


@pytest.fixture
def X():
    # The 9 combinations of c1 and c2 occur 1 to 9 times, so the frequency of c1_&_c2 tells nothing its categories do
    # not, and it is not a candidate for frequency encoding. The interactions with c3 are.
    pairs = [(a, b) for a in "abc" for b in "xyz"]
    rows = [pair for count, pair in enumerate(pairs, start=1) for _ in range(count)]
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"c1": [a for a, _ in rows], "c2": [b for _, b in rows], "c3": rng.choice(list("pq"), len(rows))}
    ).astype("category")


@pytest.mark.parametrize(
    "freq_option, expected_columns",
    [
        (
            "add_freq",
            ["c1_&_c2", "c1_&_c3", "c2_&_c3", "c1_&_c2_&_c3", "c1_&_c3_freq", "c2_&_c3_freq", "c1_&_c2_&_c3_freq"],
        ),
        ("only_freq", ["c1_&_c2", "c1_&_c3_freq", "c2_&_c3_freq", "c1_&_c2_&_c3_freq"]),
    ],
    ids=["add_freq", "only_freq"],
)
def test_categorical_interaction_frequency_encoding(X, freq_option, expected_columns):
    """`add_freq` and `only_freq` frequency-encode the new features that are candidates, and keep the others.

    `fit_transform` used to raise an AttributeError: `_fit` sets `fitted_` before the frequency encoding, which then
    took its transform branch and used a `cat_freq` that was never created. The new features that are not candidates
    were also dropped, since only the candidates were passed to the `FrequencyFeatureGenerator`.
    """
    y = pd.Series(np.arange(len(X), dtype=float))
    generator = CategoricalInteractionFeatureGenerator(target_type="regression", **{freq_option: True})

    X_fit = generator.fit_transform(X, y=y)
    X_out = generator.transform(X)

    assert list(X_out.columns) == expected_columns
    # `FrequencyFeatureGenerator.fit_transform` returns the columns it keeps twice, which is fixed separately
    pd.testing.assert_frame_equal(X_fit.loc[:, ~X_fit.columns.duplicated()], X_out)
