"""Bagged fits validate fit constraints against the full training data, not per-fold slices.

The bag's `validate_fit_args` checks itself and the child template against the full training
split; each fold child then trains on a (k-1)/k slice. Without the upstream-validation flag the
children re-validated on their slices, so a dataset whose full split satisfies `ag.min_rows` but
whose slices do not failed every child and lost the model entirely.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from autogluon.core.models import BaggedEnsembleModel
from autogluon.core.models.dummy.dummy_model import DummyModel
from autogluon.core.models.abstract.abstract_model import ConstraintViolationError


def _make_data(n: int):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n)})
    y = pd.Series(rng.choice([0, 1], size=n))
    return X, y


def _make_bag(tmp_path, min_rows: int) -> BaggedEnsembleModel:
    base = DummyModel(
        path=str(tmp_path),
        name="Dummy",
        problem_type="binary",
        eval_metric="accuracy",
        hyperparameters={"ag.min_rows": min_rows},
    )
    return BaggedEnsembleModel(
        model_base=base,
        path=str(tmp_path),
        name="Dummy_BAG",
        hyperparameters={"use_child_oof": False, "save_bag_folds": True},
    )


def test_min_rows_between_fold_and_full_size_fits(tmp_path):
    """Full split 80 >= min_rows 75 > fold slice 70: the bag accepts the model, so every fold
    child must fit rather than re-reject its slice."""
    X, y = _make_data(80)
    bag = _make_bag(tmp_path, min_rows=75)
    bag.validate_fit_args(X=X, y=y)  # what the trainer checks before fitting: must accept
    bag.fit(X=X, y=y, k_fold=8)
    assert bag.n_children == 8


def test_min_rows_above_full_size_still_rejected(tmp_path):
    """A genuine violation (full split 60 < min_rows 75) must still be rejected at the bag level,
    which is where the trainer turns it into a skip."""
    X, y = _make_data(60)
    bag = _make_bag(tmp_path, min_rows=75)
    with pytest.raises(ConstraintViolationError):
        bag.validate_fit_args(X=X, y=y)


def test_flag_survives_the_fold_copy(tmp_path):
    """Fold models are deepcopies of the model base, so the upstream-validation flag set by the
    bag must carry over to them."""
    base = DummyModel(path=str(tmp_path), name="Dummy", problem_type="binary", eval_metric="accuracy")
    assert base._fit_constraints_validated_upstream is False
    base._fit_constraints_validated_upstream = True
    assert copy.deepcopy(base)._fit_constraints_validated_upstream is True
