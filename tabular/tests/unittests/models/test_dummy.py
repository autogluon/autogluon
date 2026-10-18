from pathlib import Path

import pytest

from autogluon.core.models.dummy.dummy_model import DummyModel
from autogluon.tabular import TabularPredictor
from autogluon.tabular.testing import FitHelper, ModelFitHelper


def test_no_models_will_raise():
    """Tests that RuntimeError is raised when no models fit"""
    fit_args = dict(
        hyperparameters={},
    )

    dataset_name = "toy_binary"
    train_data, test_data, dataset_info = FitHelper.load_dataset(name=dataset_name)

    with pytest.raises(RuntimeError):
        FitHelper.fit_dataset(train_data=train_data, init_args=dict(label=dataset_info["label"]), fit_args=fit_args)


def test_no_models():
    """Tests that logic works properly when no models are trained and raise_on_no_models_fitted=False"""
    fit_args = dict(
        hyperparameters={},
        raise_on_no_models_fitted=False,
    )

    dataset_name = "toy_binary"
    train_data, test_data, dataset_info = FitHelper.load_dataset(name=dataset_name)

    predictor = FitHelper.fit_dataset(
        train_data=train_data, init_args=dict(label=dataset_info["label"]), fit_args=fit_args
    )

    assert not predictor.model_names()
    with pytest.raises(AssertionError):
        predictor.predict(test_data)
    assert len(predictor.leaderboard()) == 0
    assert len(predictor.leaderboard(test_data)) == 0
    assert len(predictor.model_failures()) == 0


def test_no_models_raise():
    """Tests that logic works properly when no models are trained, and tests predictor.model_failures() and raise_on_no_models_fitted=False"""

    expected_exc_str = "Test Error Message"

    # Force DummyModel to raise an exception when fit.
    fit_args = dict(
        hyperparameters={DummyModel: {"raise": ValueError, "raise_msg": expected_exc_str}},
        raise_on_no_models_fitted=False,
    )

    dataset_name = "toy_binary"
    train_data, test_data, dataset_info = FitHelper.load_dataset(name=dataset_name)

    predictor = FitHelper.fit_dataset(
        train_data=train_data, init_args=dict(label=dataset_info["label"]), fit_args=fit_args
    )

    assert not predictor.model_names()
    with pytest.raises(AssertionError):
        predictor.predict(test_data)
    assert len(predictor.leaderboard()) == 0
    assert len(predictor.leaderboard(test_data)) == 0

    model_failures = predictor.model_failures()
    assert len(model_failures) == 1
    model_failures_dict = model_failures.iloc[0].to_dict()
    assert model_failures_dict["model"] == "Dummy"
    assert model_failures_dict["model_type"] == "DummyModel"
    assert model_failures_dict["exc_type"] == "ValueError"
    assert model_failures_dict["exc_str"] == expected_exc_str


def test_raise_on_model_failure():
    """Tests that logic works properly when model raises exception and raise_on_model_failure=True"""

    expected_exc_str = "Test Error Message"

    train_data, test_data, dataset_info = FitHelper.load_dataset(name="toy_binary")

    # Force DummyModel to raise an exception when fit.
    fit_args = dict(
        hyperparameters={DummyModel: {"raise": ValueError, "raise_msg": expected_exc_str}},
        raise_on_model_failure=True,
        feature_generator=None,
    )

    with pytest.raises(ValueError) as excinfo:
        FitHelper.fit_dataset(train_data=train_data, init_args=dict(label=dataset_info["label"]), fit_args=fit_args)
    assert str(excinfo.value) == "Test Error Message"


def _fit_and_skip(train_data, label: str, dummy_hyperparameters: dict, **extra_fit_args):
    """Fit a constrained DummyModel next to a model that trains, and return the predictor.

    Keeps `raise_on_model_failure=True` throughout: a constraint skip must not abort a strict run,
    and pairing with a trainable model is what lets the run finish so the skip can be inspected.
    """
    return FitHelper.fit_dataset(
        train_data=train_data,
        init_args=dict(label=label),
        fit_args=dict(
            hyperparameters={DummyModel: [dummy_hyperparameters], "GBM": [{"num_boost_round": 5}]},
            raise_on_model_failure=True,
            fit_weighted_ensemble=False,
            **extra_fit_args,
        ),
    )


def _assert_skipped(predictor, expected: str) -> None:
    """The DummyModel was skipped for `expected`, and it is not counted as a failure."""
    assert not any("Dummy" in name for name in predictor.model_names()), "constrained model trained anyway"
    assert predictor._trainer._models_failed_to_train_errors == {}, "constraint skip counted as a failure"
    assert expected in predictor._trainer._models_skipped_by_constraint["Dummy"]


def test_raise_on_fit_args():
    """Tests that ag.min_rows, ag.max_rows, ag.max_features, ag.max_classes, ag.problem_types work.

    Each constraint gates the model: the fit skips it and carries on. These used to assert an
    AssertionError under `raise_on_model_failure=True`; a constraint miss is a configured outcome
    rather than breakage, so it no longer aborts a strict run, and the skip is asserted instead.
    """

    dataset_name = "toy_binary"
    train_data, test_data, dataset_info = FitHelper.load_dataset(name=dataset_name)

    _assert_skipped(_fit_and_skip(train_data, dataset_info["label"], {"ag.max_rows": 1}), "ag.max_rows=1")

    fit_args = dict(
        hyperparameters={DummyModel: [{"ag.max_rows": 1, "ag.ignore_constraints": True}]},
        raise_on_model_failure=True,
    )

    # This works because ag.ignore_constraints is set to True, bypassing `ag.max_rows`
    FitHelper.fit_dataset(train_data=train_data, init_args=dict(label=dataset_info["label"]), fit_args=fit_args)

    assert len(train_data) == 4

    fit_args = dict(
        hyperparameters={DummyModel: [{"ag.max_rows": 3}]},
        raise_on_model_failure=True,
    )

    # This works because len(X) = 3 and len(X_val) = 1
    FitHelper.fit_dataset(train_data=train_data, init_args=dict(label=dataset_info["label"]), fit_args=fit_args)

    # Check that bagging uses the full data for checking max rows. Only the DummyModel is
    # configured here: 4 folds over 4 rows is too thin for a real model to pair with, so the run
    # ends with nothing trained, which is itself the proof the constraint gated the fit.
    fit_args = dict(
        hyperparameters={DummyModel: [{"ag.max_rows": 3}]},
        raise_on_model_failure=True,
        num_bag_folds=4,
    )

    with pytest.raises(RuntimeError, match="No models were trained successfully"):
        FitHelper.fit_dataset(train_data=train_data, init_args=dict(label=dataset_info["label"]), fit_args=fit_args)

    _assert_skipped(_fit_and_skip(train_data, dataset_info["label"], {"ag.min_rows": 100}), "ag.min_rows=100")

    fit_args = dict(
        hyperparameters={DummyModel: [{"ag.min_rows": 100, "ag.ignore_constraints": True}]},
        raise_on_model_failure=True,
    )

    # This works because ag.ignore_constraints is set to True, bypassing `ag.min_rows`
    FitHelper.fit_dataset(train_data=train_data, init_args=dict(label=dataset_info["label"]), fit_args=fit_args)

    fit_args = dict(
        hyperparameters={DummyModel: [{"ag.min_rows": 2}]},
        raise_on_model_failure=True,
    )

    # This works because len(X) = 2
    FitHelper.fit_dataset(train_data=train_data, init_args=dict(label=dataset_info["label"]), fit_args=fit_args)

    _assert_skipped(_fit_and_skip(train_data, dataset_info["label"], {"ag.max_features": 0}), "ag.max_features=0")

    fit_args = dict(
        hyperparameters={DummyModel: [{"ag.max_features": 1}]},
        raise_on_model_failure=True,
    )

    # This works because len(X.columns) == 1
    FitHelper.fit_dataset(train_data=train_data, init_args=dict(label=dataset_info["label"]), fit_args=fit_args)

    _assert_skipped(_fit_and_skip(train_data, dataset_info["label"], {"ag.max_classes": 1}), "ag.max_classes=1")

    fit_args = dict(
        hyperparameters={DummyModel: [{"ag.max_classes": 2}]},
        raise_on_model_failure=True,
    )

    # This works because self.num_classes = 2
    FitHelper.fit_dataset(train_data=train_data, init_args=dict(label=dataset_info["label"]), fit_args=fit_args)

    _assert_skipped(
        _fit_and_skip(train_data, dataset_info["label"], {"ag.problem_types": ["abc", "multiclass", "regression"]}),
        "ag.problem_types=['abc', 'multiclass', 'regression']",
    )

    fit_args = dict(
        hyperparameters={DummyModel: [{"ag.problem_types": ["binary", "def"]}]},
        raise_on_model_failure=True,
    )

    # This works because self.problem_type = "binary"
    FitHelper.fit_dataset(train_data=train_data, init_args=dict(label=dataset_info["label"]), fit_args=fit_args)


def test_dummy():
    model_cls = DummyModel
    model_hyperparameters = {}

    """Additionally tests that all metrics work"""
    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters, extra_metrics=True)


def test_dummy_binary_model():
    fit_args = dict()
    dataset_name = "toy_binary"
    ModelFitHelper.fit_and_validate_dataset(dataset_name=dataset_name, model=DummyModel(), fit_args=fit_args)


def test_dummy_multiclass_model():
    fit_args = dict()
    dataset_name = "toy_multiclass"
    ModelFitHelper.fit_and_validate_dataset(dataset_name=dataset_name, model=DummyModel(), fit_args=fit_args)


def test_dummy_regression_model():
    fit_args = dict()
    dataset_name = "toy_regression"
    ModelFitHelper.fit_and_validate_dataset(dataset_name=dataset_name, model=DummyModel(), fit_args=fit_args)


def test_dummy_binary_absolute_path():
    """Test that absolute path works"""
    fit_args = dict(
        hyperparameters={DummyModel: {}},
    )
    path = Path(".") / "AG_test"
    path = str(path.resolve())
    init_args = dict(path=path)

    dataset_name = "toy_binary"

    FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, init_args=init_args, fit_args=fit_args)


def test_dummy_binary_absolute_path_stack():
    """Test that absolute path works"""
    fit_args = dict(
        hyperparameters={DummyModel: {}},
        num_bag_folds=2,
        num_bag_sets=2,
        num_stack_levels=1,
    )

    dataset_name = "toy_binary"
    FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name, fit_args=fit_args, expected_model_count=4, path_as_absolute=True
    )


def test_dummy_binary_model_absolute_path():
    """Test that absolute path works"""
    fit_args = dict()
    path = Path(".") / "AG_test"
    path = str(path.resolve())
    model = DummyModel(path=path)
    dataset_name = "toy_binary"
    ModelFitHelper.fit_and_validate_dataset(dataset_name=dataset_name, model=model, fit_args=fit_args)


def test_dummy_ag_ens_hyperparameter():
    """
    Verifies that sending ag_args_ensemble arguments via the `ag.ens.` prefix works.
    """
    hyperparameters = {
        "ag.ens.fold_fitting_strategy": "sequential_local",
        "ag.ens.foo": "bar",
        "key1": "val1",
    }
    fit_args = dict(
        hyperparameters={DummyModel: hyperparameters},
        num_bag_folds=2,
    )
    dataset_name = "toy_binary"

    predictor: TabularPredictor = FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        delete_directory=False,
        refit_full=False,
        fit_weighted_ensemble=False,
    )
    assert len(predictor.model_names()) == 1
    model_name = predictor.model_names()[0]
    model_info = predictor.model_info(model=model_name)
    assert model_info["hyperparameters_user"]["fold_fitting_strategy"] == "sequential_local"
    assert model_info["hyperparameters_user"]["foo"] == "bar"
    assert model_info["hyperparameters"]["fold_fitting_strategy"] == "sequential_local"
    assert model_info["hyperparameters"]["foo"] == "bar"
    assert "key1" not in model_info["hyperparameters"]
    assert model_info["bagged_info"]["child_hyperparameters"] == {"key1": "val1"}


def test_constraint_violation_is_a_clean_skip():
    """A constraint miss is a configuration outcome, so it must skip cleanly, not look like a crash.

    The trainer classifies exceptions by type to decide whether to print a one-line skip or a
    failure with a traceback. Constraints raised a bare `AssertionError`, which fell through to
    the failure branch, so a benign, expected skip (e.g. a foundation model past its row limit)
    printed a traceback and read as a crash.
    """
    from autogluon.core.utils.exceptions import ConstraintViolationError

    # Still an AssertionError, which is what these constraints raised before this exception existed,
    # so anything catching that keeps working.
    assert issubclass(ConstraintViolationError, AssertionError)

    # Two message forms: the standalone one names the model, while `reason` omits it because the
    # trainer's skip line already does ("Skipping Dummy because a fit constraint ...: <reason>.").
    exc = ConstraintViolationError(
        "ag.max_rows=1 for model 'Dummy', but found 2 rows.",
        reason="ag.max_rows=1, but the data has 2 rows",
    )
    assert "Dummy" in str(exc)
    assert "Dummy" not in exc.reason

    # End to end: the model is skipped, and the recorded message is the model-naming form.
    train_data, _, dataset_info = FitHelper.load_dataset(name="toy_binary")
    predictor = _fit_and_skip(train_data, dataset_info["label"], {"ag.max_rows": 1})
    _assert_skipped(predictor, "ag.max_rows=1")
    assert "Dummy" in predictor._trainer._models_skipped_by_constraint["Dummy"]


def test_constraint_violation_is_not_a_model_failure():
    """A constraint skip must not be reported as a failed model, nor trip a strict run.

    The model was configured not to run on data like this, so counting it among the failures
    would make a healthy run look broken. `raise_on_model_failure=True` still aborts (see
    `test_raise_on_fit_args`) -- that flag is about stopping at the first model that does not fit,
    which is a separate question from whether the skip is a failure.
    """
    train_data, _, dataset_info = FitHelper.load_dataset(name="toy_binary")
    init_args = dict(label=dataset_info["label"])
    hyperparameters = {DummyModel: [{"ag.max_rows": 1}], "GBM": [{"num_boost_round": 5}]}

    predictor = FitHelper.fit_dataset(
        train_data=train_data,
        init_args=init_args,
        fit_args=dict(hyperparameters=hyperparameters, fit_weighted_ensemble=False),
    )
    trained = predictor.model_names()
    assert not any("Dummy" in name for name in trained)
    assert any("LightGBM" in name for name in trained)
    # Not a failure, but still introspectable.
    assert predictor._trainer._models_failed_to_train_errors == {}
    assert "ag.max_rows=1" in predictor._trainer._models_skipped_by_constraint["Dummy"]


def test_constraint_violation_still_raises_when_nothing_is_trained():
    """Skipping is fine; skipping *everything* is not, and must still surface."""
    train_data, _, dataset_info = FitHelper.load_dataset(name="toy_binary")

    with pytest.raises(RuntimeError, match="No models were trained successfully"):
        FitHelper.fit_dataset(
            train_data=train_data,
            init_args=dict(label=dataset_info["label"]),
            fit_args=dict(hyperparameters={DummyModel: [{"ag.max_rows": 1}]}, fit_weighted_ensemble=False),
        )


def test_cell_and_feature_bounds():
    """`ag.min_features` / `max_features` / `min_cells` / `max_cells` gate the fit.

    Cell bounds exist because row and feature bounds cannot express total table size: a feature
    limit wide enough for a short-and-wide table also admits a long-and-wide one many times
    larger. The last case below passes generous bounds on each axis alone yet exceeds the cells.

    `toy_binary` is 4 rows x 1 feature, and the model fits on 2 rows with 2 held out for
    validation, so the bounds are checked against 2 rows x 1 feature = 2 cells.
    """

    train_data, _, dataset_info = FitHelper.load_dataset(name="toy_binary")
    init_args = dict(label=dataset_info["label"])

    label = dataset_info["label"]

    def _fit_alone(hyperparameters: dict):
        FitHelper.fit_dataset(
            train_data=train_data,
            init_args=init_args,
            fit_args=dict(hyperparameters={DummyModel: [hyperparameters]}, raise_on_model_failure=True),
        )

    # Satisfied bounds train fine.
    _fit_alone({"ag.min_features": 1, "ag.max_features": 1})
    _fit_alone({"ag.min_cells": 1, "ag.max_cells": 2})
    # ignore_constraints covers the new bounds too.
    _fit_alone({"ag.max_cells": 1, "ag.ignore_constraints": True})

    _assert_skipped(_fit_and_skip(train_data, label, {"ag.min_features": 100}), "ag.min_features=100")
    _assert_skipped(_fit_and_skip(train_data, label, {"ag.min_cells": 10_000}), "ag.min_cells=10000")
    # The message reports the shape that produced the count, not just the total.
    _assert_skipped(_fit_and_skip(train_data, label, {"ag.max_cells": 1}), "2 cells (2 rows x 1 features)")
    # Generous row and feature bounds, exceeded cell budget: what cell bounds are for.
    _assert_skipped(
        _fit_and_skip(train_data, label, {"ag.max_rows": 1000, "ag.max_features": 1000, "ag.max_cells": 1}),
        "ag.max_cells=1",
    )
