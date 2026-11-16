import numpy as np
import pandas as pd

from autogluon.core.metrics import root_mean_squared_error
from autogluon.core.utils.utils import compute_permutation_feature_importance, shuffle_df_rows


def _frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": np.arange(n), "b": rng.normal(size=n)}, index=np.arange(n) + 1000)


def test_shuffle_df_rows_is_a_permutation():
    """Every row appears exactly once, moved whole, and the same seed gives the same order.

    Rows used to be drawn with replacement, which dropped about a third of them and repeated others.
    """
    X = _frame()

    X_shuffled = shuffle_df_rows(X, seed=3)

    assert sorted(X_shuffled["a"]) == list(X["a"])
    assert not X_shuffled["a"].equals(X["a"].reset_index(drop=True))
    assert np.array_equal(X_shuffled["b"].to_numpy(), X.set_index("a").loc[X_shuffled["a"], "b"].to_numpy())
    assert list(X_shuffled.index) == list(range(len(X)))
    assert X_shuffled.equals(shuffle_df_rows(X, seed=3))


def test_permutation_importance_leaves_the_global_random_state_untouched():
    """The shuffles use their own generator, so the caller's `np.random` sequence is not reset."""
    X = _frame()
    y = 2 * X["b"]
    np.random.seed(123)
    expected = np.random.rand(3)

    np.random.seed(123)
    compute_permutation_feature_importance(
        X=X,
        y=y,
        predict_func=lambda df: 2 * df["b"].to_numpy(),
        eval_metric=root_mean_squared_error,
        num_shuffle_sets=3,
        silent=True,
    )

    assert np.array_equal(np.random.rand(3), expected)
