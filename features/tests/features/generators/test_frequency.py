import numpy as np
import pandas as pd
import pytest

from autogluon.features.generators import FrequencyFeatureGenerator


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "n": rng.normal(size=n),
            "c1": pd.Series(rng.choice(list("abc"), n, p=[0.6, 0.3, 0.1]), dtype="category"),
            "c2": pd.Series(rng.choice(list("xyz"), n), dtype="category"),
        }
    )


@pytest.mark.parametrize(
    "keep_original, expected_columns",
    [(True, ["n", "c1", "c2", "c1_freq", "c2_freq"]), (False, ["n", "c1_freq", "c2_freq"])],
    ids=["keep_original", "drop_original"],
)
def test_frequency_fit_transform_matches_transform(X, keep_original, expected_columns):
    """`fit_transform` returns the columns that `transform` returns for the same data, each once.

    With `keep_original=True` it used to return every input column twice.
    """
    generator = FrequencyFeatureGenerator(keep_original=keep_original)

    X_out = generator.fit_transform(X)

    assert list(X_out.columns) == expected_columns
    pd.testing.assert_frame_equal(generator.transform(X), X_out)
    assert (X_out["c1_freq"] == X["c1"].map(X["c1"].value_counts()).astype(float)).all()
