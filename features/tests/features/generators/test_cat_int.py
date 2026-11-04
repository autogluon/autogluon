import numpy as np
import pandas as pd

from autogluon.features.generators import CategoricalInteractionFeatureGenerator


def _categorical_frame(n_columns: int, n_rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {f"c{i}": pd.Series(rng.choice(list("abcdef"), n_rows), dtype="category") for i in range(1, n_columns + 1)}
    )


def test_categorical_interaction_with_two_categorical_columns():
    """With exactly two categorical columns, their pairwise interaction is generated.

    No interaction used to be generated when there were fewer categorical columns than `max_order` (3 by default).
    """
    X = _categorical_frame(n_columns=2)
    y = pd.Series(np.arange(len(X), dtype=float))
    generator = CategoricalInteractionFeatureGenerator(target_type="regression")

    X_out = generator.fit_transform(X, y=y)

    assert list(X_out.columns) == ["c1_&_c2"]
    pd.testing.assert_frame_equal(generator.transform(X), X_out)
    # one category per combination of values seen in training
    assert X_out["c1_&_c2"].nunique() == len(X.drop_duplicates())


def test_categorical_interaction_with_three_categorical_columns():
    """With three categorical columns, all pairwise interactions and the three-way one are generated."""
    X = _categorical_frame(n_columns=3)
    y = pd.Series(np.arange(len(X), dtype=float))

    X_out = CategoricalInteractionFeatureGenerator(target_type="regression").fit_transform(X, y=y)

    assert sorted(X_out.columns) == sorted(["c1_&_c2", "c1_&_c3", "c2_&_c3", "c1_&_c2_&_c3"])
