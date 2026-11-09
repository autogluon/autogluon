from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from autogluon.core.models import BaggedEnsembleModel, DummyModel
from autogluon.tabular import TabularPredictor
from autogluon.tabular.trainer.abstract_trainer import AbstractTabularTrainer


@pytest.mark.parametrize(
    "fit_kwargs, parts",
    [
        ({}, ["part-000.pkl"]),  # every set in one fit call, saved once
        ({"time_limit": 600, "delay_bag_sets": True}, ["part-000.pkl", "part-001.pkl"]),  # one set per call
    ],
)
def test_oof_per_fold_through_the_predictor(tmp_path, fit_kwargs, parts):
    """`ag.ens.oof_per_fold` reaches the bag through the predictor, which keeps every fold model's out-of-fold
    predictions whether it fits a repeated bag in one call or one set per call (a part file per save that adds
    folds)."""
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    train["label"] = (train["a"] + rng.normal(scale=0.8, size=200) > 0).astype(int)
    predictor = TabularPredictor(label="label", path=str(tmp_path), verbosity=0).fit(
        train,
        hyperparameters={"LR": {"ag.ens.oof_per_fold": True}},
        num_bag_folds=3,
        num_bag_sets=2,
        fit_weighted_ensemble=False,
        **fit_kwargs,
    )
    trainer = predictor._trainer
    (name,) = trainer.get_model_names()
    bag = trainer.load_model(name)

    per_fold = bag.predict_proba_oof_per_fold()
    assert len(per_fold) == 6
    total, count = np.zeros(len(train)), np.zeros(len(train))
    for repeat in range(2):
        rows = np.concatenate([rows for rows, _ in per_fold[repeat * 3 : (repeat + 1) * 3]])
        np.testing.assert_array_equal(np.sort(rows), np.arange(len(train)))
    for rows, pred in per_fold:
        total[rows] += pred
        count[rows] += 1
    np.testing.assert_allclose(trainer.get_model_oof(name), total / count, rtol=1e-5)
    utils = os.path.join(bag.path, "utils")
    assert sorted(os.listdir(os.path.join(utils, "oof_per_fold"))) == parts
    assert not os.path.exists(os.path.join(utils, "oof.pkl")), "no pooled file: derived from the folds"


def test_restoring_the_oof_alignment_moves_each_folds_rows():
    """A model fit on some rows has its out-of-fold predictions scattered onto the full training frame; each fold's
    rows must move with them."""
    rng = np.random.default_rng(0)
    covered_rows = pd.Series(np.arange(60) % 3 != 0)  # the model saw 40 of 60 rows
    X = pd.DataFrame({"a": rng.normal(size=int(covered_rows.sum()))})
    y = pd.Series((X["a"] > 0).astype(int))
    bag = BaggedEnsembleModel(
        model_base=DummyModel(problem_type="binary"),
        hyperparameters={"fold_fitting_strategy": "sequential_local", "oof_per_fold": True},
    ).fit(X=X, y=y, k_fold=4)
    before = bag.predict_proba_oof_per_fold()

    trainer = SimpleNamespace(load_model=lambda model_name: bag, save_to_disk=False)
    AbstractTabularTrainer._restore_oof_alignment(trainer, "bag", covered_rows=covered_rows)

    covered_idx = np.flatnonzero(covered_rows)
    assert bag._oof_per_fold["rows"].dtype == np.uint8, "stored compactly (60 rows)"
    assert bag._oof_per_fold["n_rows"] == 60
    assert bag._oof_per_fold_saved is None, "every fold's rows changed: saved again in full"
    after = bag.predict_proba_oof_per_fold()
    for (rows_before, pred_before), (rows_after, pred_after) in zip(before, after):
        np.testing.assert_array_equal(rows_after, covered_idx[rows_before])
        np.testing.assert_array_equal(pred_after, pred_before)
        np.testing.assert_allclose(bag.predict_proba_oof()[rows_after], pred_after, rtol=1e-6)
