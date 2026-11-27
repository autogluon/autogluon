import numpy as np
import pandas as pd
import pytest

from autogluon.common.utils.validation_structure import ValidationStructure


@pytest.mark.parametrize(
    "a,b",
    [
        (["a||b", "a", "a||b", "a"], ["c", "b||c", "c", "b||c"]),
        ([1, "1", 1, "1"], ["x", "x", "x", "x"]),
    ],
)
def test_composite_group_keys_preserve_distinct_groups(a, b):
    X = pd.DataFrame({"a": a, "b": b}, index=[10, 20, 30, 40])
    y = pd.Series([0.0, 1.0, 0.0, 1.0], index=X.index)
    structure = ValidationStructure(group_on=["a", "b"])
    assert structure.num_group_instances(X) == 2

    splits, folds, repeats = structure.custom_splits(X, y, num_folds=2, problem_type="regression")
    assert (folds, repeats) == (2, 1)
    assert {tuple(val_idx) for _, val_idx in splits} == {(0, 2), (1, 3)}
    for train_idx, val_idx in splits:
        np.testing.assert_array_equal(np.sort(np.concatenate([train_idx, val_idx])), np.arange(len(X)))

    train_idx, val_idx = structure.holdout_split_indices(X, y, holdout_frac=0.5, problem_type="regression")
    assert {tuple(train_idx), tuple(val_idx)} == {(0, 2), (1, 3)}
