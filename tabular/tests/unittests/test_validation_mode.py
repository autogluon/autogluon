"""`validation_mode="none"`: train on every row, combine with predetermined weights.

The motivating case is an in-context learner such as TabPFN or TabICL, which needs no validation
set, paired with an ensemble combination that is known up front rather than learned from held-out
predictions. These tests use fast models so they exercise the plumbing, not the backends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autogluon.tabular import TabularPredictor

HYPERPARAMETERS = {"GBM": {}, "XGB": {}}
WEIGHTS = {"LightGBM": 0.5, "XGBoost": 0.5}


def _data(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    df["label"] = (df.f0 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return df


def _fit(tmp_path, train, **kwargs):
    return TabularPredictor(label="label", path=str(tmp_path), verbosity=0).fit(
        train, hyperparameters=HYPERPARAMETERS, **kwargs
    )


def test_validation_mode_none_trains_on_every_row(tmp_path):
    """No holdout is carved, and no model gets a validation score."""
    train = _data()
    predictor = _fit(tmp_path, train, validation_mode="none", ensemble_weights=WEIGHTS)

    assert predictor._trainer._num_rows_train == len(train)
    leaderboard = predictor.leaderboard(silent=True)
    assert leaderboard["score_val"].isna().all()
    assert predictor.model_best == "WeightedEnsemble_L2"


def test_validation_mode_none_applies_the_given_weights(tmp_path):
    """The ensemble is exactly the weighted combination of its base models."""
    train = _data()
    test = _data(n=100, seed=1).drop(columns=["label"])
    weights = {"LightGBM": 0.25, "XGBoost": 0.75}
    predictor = _fit(tmp_path, train, validation_mode="none", ensemble_weights=weights)

    ensemble = predictor.predict_proba(test)[1].to_numpy()
    expected = sum(w * predictor.predict_proba(test, model=name)[1].to_numpy() for name, w in weights.items())
    np.testing.assert_allclose(ensemble, expected, atol=1e-6)


def test_validation_mode_none_normalizes_weights(tmp_path):
    """Weights need not sum to 1; the ratio is what matters."""
    train = _data()
    test = _data(n=60, seed=2).drop(columns=["label"])
    scaled = _fit(tmp_path / "a", train, validation_mode="none", ensemble_weights={"LightGBM": 2, "XGBoost": 6})
    unit = _fit(tmp_path / "b", train, validation_mode="none", ensemble_weights={"LightGBM": 0.25, "XGBoost": 0.75})

    np.testing.assert_allclose(
        scaled.predict_proba(test)[1].to_numpy(), unit.predict_proba(test)[1].to_numpy(), atol=1e-6
    )


def test_validation_mode_none_single_model_without_an_ensemble(tmp_path):
    """One model and no score is unambiguous: it is the only thing `predict` could mean."""
    train = _data()
    predictor = TabularPredictor(label="label", path=str(tmp_path), verbosity=0).fit(
        train, hyperparameters={"GBM": {}}, validation_mode="none", fit_weighted_ensemble=False
    )

    assert predictor._trainer._num_rows_train == len(train)
    assert predictor.model_best == "LightGBM"
    assert len(predictor.predict(train.drop(columns=["label"]))) == len(train)


def test_validation_mode_none_multiple_models_without_an_ensemble(tmp_path):
    """Several models, no scores and no weights: there is no basis for a default.

    Answering `predict` from whichever model happened to be last in the DAG would be an arbitrary
    choice the caller never made, so the ambiguity is reported instead. The models themselves are
    fit and usable by name.
    """
    from autogluon.tabular.trainer.abstract_trainer import AmbiguousModelBestError

    train = _data()
    test = _data(n=50, seed=3).drop(columns=["label"])
    predictor = _fit(tmp_path, train, validation_mode="none", fit_weighted_ensemble=False)

    assert set(predictor.model_names()) == {"LightGBM", "XGBoost"}
    assert predictor.model_best is None, "no model should be designated best without a basis"
    with pytest.raises(AmbiguousModelBestError, match="no basis to choose between"):
        predictor.predict(test)
    # Naming a model is the documented way out, and it works.
    assert len(predictor.predict(test, model="LightGBM")) == len(test)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        # Reported by `resolve_validation_mode`, which sees the resolved counts.
        ({"num_bag_folds": 3}, "cannot be combined with num_bag_folds=3"),
        ({"tuning_data": "TRAIN"}, "cannot be combined with `tuning_data`"),
        ({"validation_structure": {"group_on": "f0"}}, "cannot be combined with `validation_structure`"),
    ],
)
def test_validation_mode_none_rejects_what_needs_a_holdout(tmp_path, kwargs, match):
    """Bagging, stacking and explicit tuning data are all defined by holding rows out."""
    train = _data()
    if kwargs.get("tuning_data") == "TRAIN":
        kwargs["tuning_data"] = train
    with pytest.raises(ValueError, match=match):
        _fit(tmp_path, train, validation_mode="none", ensemble_weights=WEIGHTS, **kwargs)


def test_validation_mode_none_requires_weights_for_an_ensemble(tmp_path):
    """Without validation data there is nothing to learn weights from, so they must be supplied."""
    with pytest.raises(ValueError, match="leaves no data to learn ensemble weights from"):
        _fit(tmp_path, _data(), validation_mode="none")


def test_ensemble_weights_rejected_when_validation_exists(tmp_path):
    """With validation data AutoGluon learns the weights; fixed weights would silently override."""
    with pytest.raises(ValueError, match="only supported with validation_mode='none'"):
        _fit(tmp_path, _data(), ensemble_weights=WEIGHTS)


def test_validation_mode_rejects_unknown_values(tmp_path):
    with pytest.raises(ValueError, match="validation_mode must be 'auto' or 'none'"):
        _fit(tmp_path, _data(), validation_mode="sometimes")


@pytest.mark.parametrize(
    "weights, match",
    [
        # Caught before fitting -- no requested model could produce this name.
        ({"Nope": 1.0}, "do not match any requested model"),
        # Caught before fitting: XGBoost is requested but has no weight.
        ({"LightGBM": 1.0}, "gives no weight to"),
        ({"LightGBM": 0.0, "XGBoost": 0.0}, "must sum to a positive value"),
    ],
)
def test_ensemble_weights_are_validated(tmp_path, weights, match):
    """A silent mismatch would give a model someone else's weight."""
    with pytest.raises(ValueError, match=match):
        _fit(tmp_path, _data(), validation_mode="none", ensemble_weights=weights)


def test_ensemble_weight_names_are_checked_before_fitting(tmp_path):
    """A name no requested model could produce fails before any model is trained.

    The trainer checks names again against the models that actually fitted, but that happens after
    every base model is trained -- for in-context models that is the expensive part.
    """
    train = _data()
    with pytest.raises(ValueError, match="do not match any requested model"):
        _fit(tmp_path, train, validation_mode="none", ensemble_weights={"LightGBM": 0.5, "CatBoost": 0.5})
    assert not (tmp_path / "models").exists(), "no model should have been fit"


def test_ensemble_weights_hint_at_hyperparameters_keys(tmp_path):
    """`hyperparameters` is keyed by 'GBM'; ensemble_weights wants 'LightGBM'.

    Reaching for the key is the predictable mistake, so the error names the mapping.
    """
    with pytest.raises(ValueError, match="look like `hyperparameters` keys"):
        _fit(tmp_path, _data(), validation_mode="none", ensemble_weights={"GBM": 0.5, "XGB": 0.5})


def test_ensemble_weight_name_extending_a_real_model_is_left_to_the_trainer(tmp_path):
    """`name_suffix` concatenates without a separator, so 'LightGBMm' could have been legitimate.

    Rejecting it up front would risk refusing a real suffixed name, so it is caught after fitting
    instead -- with the exact list of models that were actually produced.
    """
    with pytest.raises(ValueError, match="are not fitted models"):
        _fit(tmp_path, _data(), validation_mode="none", ensemble_weights={"LightGBMm": 0.5, "XGBoost": 0.5})


def test_validation_mode_none_with_two_models_of_the_same_type(tmp_path):
    """Duplicate configs of one model type: 'Dummy' and 'Dummy_2'.

    Regression test. The stacker prunes its base models by validation score before fitting, which
    only compares anything when two models share a type -- and every score is None here, so the
    comparison raised and the ensemble was silently skipped. Pruning is now disabled for fixed
    weights, which it has to be anyway: dropping a base model would hand the survivors weights
    meant for other models.
    """
    train = _data()
    predictor = TabularPredictor(label="label", path=str(tmp_path), verbosity=0).fit(
        train,
        hyperparameters={"DUMMY": [{}, {}]},
        validation_mode="none",
        ensemble_weights={"Dummy": 0.5, "Dummy_2": 0.5},
    )

    assert set(predictor.model_names()) == {"Dummy", "Dummy_2", "WeightedEnsemble_L2"}
    assert predictor.model_best == "WeightedEnsemble_L2"


def test_validation_mode_none_weights_same_type_models_independently(tmp_path):
    """Same type, different configs, different weights: each must get its own.

    Uses configs that actually predict differently, so a mix-up would show up in the output rather
    than being masked by identical base predictions.
    """
    train = _data(n=250)
    test = _data(n=60, seed=4).drop(columns=["label"])
    weights = {"LightGBM": 0.2, "LightGBM_2": 0.3, "LightGBM_3": 0.5}
    predictor = TabularPredictor(label="label", path=str(tmp_path), verbosity=0).fit(
        train,
        hyperparameters={"GBM": [{"num_leaves": 4}, {"num_leaves": 20}, {"num_leaves": 60}]},
        validation_mode="none",
        ensemble_weights=weights,
    )

    base_names = [m for m in predictor.model_names() if not m.startswith("WeightedEnsemble")]
    assert sorted(base_names) == sorted(weights), "no base model may be pruned away"

    ensemble = predictor.predict_proba(test)[1].to_numpy()
    expected = sum(w * predictor.predict_proba(test, model=m)[1].to_numpy() for m, w in weights.items())
    np.testing.assert_allclose(ensemble, expected, atol=1e-6)


def test_ensemble_weights_missing_renormalize_rescales_over_survivors(tmp_path):
    """A named model that was not fit is dropped and the rest keep their proportions."""
    train = _data(n=250)
    test = _data(n=60, seed=5).drop(columns=["label"])
    predictor = TabularPredictor(label="label", path=str(tmp_path), verbosity=0).fit(
        train,
        hyperparameters={"GBM": [{"num_leaves": 5}, {"num_leaves": 40}]},
        validation_mode="none",
        ensemble_weights={"LightGBM": 0.2, "LightGBM_2": 0.3, "CatBoost": 0.5},
        ensemble_weights_missing="renormalize",
    )

    base_names = [m for m in predictor.model_names() if not m.startswith("WeightedEnsemble")]
    assert sorted(base_names) == ["LightGBM", "LightGBM_2"]
    # 0.2 / 0.3 rescaled over their own sum, not over the original 1.0.
    ensemble = predictor.predict_proba(test)[1].to_numpy()
    expected = (
        0.4 * predictor.predict_proba(test, model="LightGBM")[1].to_numpy()
        + 0.6 * predictor.predict_proba(test, model="LightGBM_2")[1].to_numpy()
    )
    np.testing.assert_allclose(ensemble, expected, atol=1e-6)


def test_ensemble_weights_missing_renormalize_handles_a_model_that_failed_to_fit(tmp_path):
    """The motivating case: the model was requested, but never produced.

    `ag.max_rows` constrains XGBoost out, so it is a genuine fit failure rather than a bad name.
    """
    train = _data(n=250)
    test = _data(n=60, seed=6).drop(columns=["label"])
    predictor = TabularPredictor(label="label", path=str(tmp_path), verbosity=0).fit(
        train,
        hyperparameters={"GBM": {}, "XGB": {"ag.max_rows": 10}},
        validation_mode="none",
        ensemble_weights={"LightGBM": 0.25, "XGBoost": 0.75},
        ensemble_weights_missing="renormalize",
    )

    assert [m for m in predictor.model_names() if not m.startswith("WeightedEnsemble")] == ["LightGBM"]
    # The sole survivor absorbs the whole weight, so the ensemble is that model.
    np.testing.assert_allclose(
        predictor.predict_proba(test)[1].to_numpy(),
        predictor.predict_proba(test, model="LightGBM")[1].to_numpy(),
        atol=1e-6,
    )


def test_ensemble_weights_missing_renormalize_still_errors_when_none_were_fit(tmp_path):
    """Dropping every named model would leave nothing to ensemble."""
    with pytest.raises(ValueError, match="None of the models named in ensemble_weights were fit"):
        TabularPredictor(label="label", path=str(tmp_path), verbosity=0).fit(
            _data(),
            hyperparameters={"GBM": {}},
            validation_mode="none",
            ensemble_weights={"CatBoost": 0.5, "RandomForest": 0.5},
            ensemble_weights_missing="renormalize",
        )


def test_ensemble_weights_missing_rejects_unknown_values(tmp_path):
    with pytest.raises(ValueError, match="ensemble_weights_missing must be 'error' or 'renormalize'"):
        _fit(tmp_path, _data(), validation_mode="none", ensemble_weights=WEIGHTS, ensemble_weights_missing="skip")


def test_hyperparameters_curve_switches_the_portfolio_with_the_mode(tmp_path):
    """Tiny data gets a small portfolio and fixed weights; larger data the full one, bagged."""
    curves = {
        "validation_mode": [[100, "none"], "auto"],
        "num_bag_folds": [[100, 0], 8],
        "num_stack_levels": [[100, 0], 1],
        "ensemble_weights": [[100, {"LightGBM": 0.5, "XGBoost": 0.5}], None],
        "hyperparameters": [[100, {"GBM": {}, "XGB": {}}], {"GBM": {}, "XGB": {}, "RF": {}}],
    }
    tiny = TabularPredictor(label="label", path=str(tmp_path / "tiny"), verbosity=0).fit(
        _data(n=60), validation_size_curves=curves
    )
    assert tiny._trainer._num_rows_train == 60
    assert tiny.leaderboard(silent=True)["score_val"].isna().all()
    assert {m for m in tiny.model_names() if not m.startswith("WeightedEnsemble")} == {"LightGBM", "XGBoost"}

    big = TabularPredictor(label="label", path=str(tmp_path / "big"), verbosity=0).fit(
        _data(n=300), validation_size_curves=curves
    )
    assert not big.leaderboard(silent=True)["score_val"].isna().all()
    assert any("RandomForest" in m for m in big.model_names())


def test_weights_and_hyperparameters_must_agree_before_fitting(tmp_path):
    """A pairing that agrees above a threshold and not below is caught pre-fit.

    Without this the config looks fine, works on larger data, and fails only in the band where the
    curves disagree -- after every base model has been trained.
    """
    curves = {
        "validation_mode": [[100, "none"], "auto"],
        "num_bag_folds": [[100, 0], 8],
        "num_stack_levels": [[100, 0], 1],
        "ensemble_weights": [[100, {"LightGBM": 0.5, "XGBoost": 0.5}], None],
        "hyperparameters": [[100, {"GBM": {}, "XGB": {}, "RF": {}}], {"GBM": {}}],
    }
    with pytest.raises(ValueError, match=r"gives no weight to \['RandomForest'\]"):
        TabularPredictor(label="label", path=str(tmp_path / "a"), verbosity=0).fit(
            _data(n=60), validation_size_curves=curves
        )
    assert not (tmp_path / "a" / "models").exists(), "no model should have been fit"
