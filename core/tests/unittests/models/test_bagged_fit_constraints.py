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
from autogluon.core.models.abstract.abstract_model import ConstraintViolationError
from autogluon.core.models.dummy.dummy_model import DummyModel


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


def _validate_with_estimate(tmp_path, estimate_n_features, max_features, estimate_raises=False):
    """Run `validate_fit_args` on an uninitialized model whose model-specific preprocessing
    changes the feature count, with the post-preprocessing estimate mocked."""
    from unittest import mock

    from autogluon.common.features.feature_metadata import FeatureMetadata

    X, y = _make_data(80)
    model = DummyModel(
        path=str(tmp_path),
        name="Dummy",
        problem_type="binary",
        eval_metric="accuracy",
        hyperparameters={"ag.max_features": max_features},
    )
    estimated = mock.MagicMock()
    estimated.get_features.return_value = [f"f{i}" for i in range(estimate_n_features)]
    with (
        mock.patch.object(model, "get_preprocessor", return_value=mock.MagicMock()),
        mock.patch.object(
            model,
            "_estimate_dtypes_after_preprocessing_cheap",
            side_effect=RuntimeError("estimate failed") if estimate_raises else None,
            return_value=None if estimate_raises else estimated,
        ),
    ):
        model.validate_fit_args(X=X, y=y, feature_metadata=FeatureMetadata.from_df(X))


def test_feature_constraints_use_post_preprocessing_count_before_initialization(tmp_path):
    """The bag validates its child template uninitialized. The feature checks must still see the
    post-preprocessing feature count, not the raw one, for models whose model-specific
    preprocessing changes it (e.g. TabPrep): raw 1 feature passes max_features=20, but the
    estimated 56 must be rejected."""
    with pytest.raises(ConstraintViolationError, match="found 56 features"):
        _validate_with_estimate(tmp_path, estimate_n_features=56, max_features=20)


def test_feature_constraint_estimate_failure_falls_back_to_raw_count(tmp_path):
    """A failing post-preprocessing estimate degrades the feature checks to the raw count rather
    than failing the model at validation time."""
    _validate_with_estimate(tmp_path, estimate_n_features=0, max_features=20, estimate_raises=True)


def test_preprocessor_construction_resolves_the_seed_sentinel(tmp_path):
    """Before `init_random_seed` runs, `random_seed` holds the "NOTSET" sentinel. Generators built
    for pre-fit constraint validation must receive the class default seed instead: an sklearn
    splitter inside a generator (e.g. out-of-fold target encoding) raises on a non-int seed, which
    previously crashed the post-preprocessing feature estimate."""
    from autogluon.features.generators import IdentityFeatureGenerator

    model = DummyModel(path=str(tmp_path), name="Dummy", problem_type="binary", eval_metric="accuracy")
    assert model.random_seed == "NOTSET"
    generator = model._init_preprocessor(preprocessor_cls=IdentityFeatureGenerator, init_params={})
    assert generator.random_state == model.default_random_seed
