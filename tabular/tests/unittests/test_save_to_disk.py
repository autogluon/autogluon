from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from autogluon.core.models.dummy.dummy_model import DummyModel
from autogluon.tabular import TabularPredictor


def _frame(problem_type: str, n: int = 200) -> pd.DataFrame:
    if problem_type == "regression":
        X, y = make_regression(n_samples=n, n_features=6, random_state=0)
    else:
        n_classes = 2 if problem_type == "binary" else 3
        X, y = make_classification(n_samples=n, n_features=6, n_informative=4, n_classes=n_classes, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["y"] = y
    return df


def _files_under(path: str) -> list[str]:
    return [os.path.join(root, f) for root, _, files in os.walk(path) for f in files]


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_in_memory_predictor_writes_nothing_and_stays_usable(problem_type):
    df = _frame(problem_type)
    train, test = df.iloc[:150], df.iloc[150:]
    path = tempfile.mkdtemp(prefix="ag_in_memory_")
    predictor = TabularPredictor(label="y", problem_type=problem_type, path=path, verbosity=0, save_to_disk=False)
    predictor.fit(train, hyperparameters={DummyModel: {}}, validation_mode="none", fit_weighted_ensemble=False)

    assert _files_under(path) == []
    assert predictor._learner.save_to_disk is False and predictor._trainer.save_to_disk is False
    assert predictor._trainer.low_memory is False and predictor.model_best in predictor._trainer.models

    pred = predictor.predict(test)
    assert len(pred) == len(test)
    if problem_type != "regression":
        proba = predictor.predict_proba(test)
        assert proba.shape[0] == len(test)
    assert len(predictor.leaderboard(test)) == 1
    info = predictor.info()
    assert info["model_info"][predictor.model_best]["memory_size"] > 0
    # persisting is a no-op that never drops the models: they exist nowhere else
    predictor.persist()
    assert predictor.unpersist() == []
    assert len(predictor.predict(test)) == len(test)
    predictor.save()
    assert _files_under(path) == []


def test_in_memory_holdout_fit_with_weighted_ensemble_writes_nothing():
    df = _frame("binary")
    train, test = df.iloc[:150], df.iloc[150:]
    predictor = TabularPredictor(
        label="y", path=tempfile.mkdtemp(prefix="ag_in_memory_"), verbosity=0, save_to_disk=False
    )
    predictor.fit(train, hyperparameters={DummyModel: {}, "GBM": {}})
    assert _files_under(predictor.path) == []
    board = predictor.leaderboard(test)
    assert "WeightedEnsemble_L2" in set(board["model"])
    assert board["score_val"].notna().all()
    predictor.delete_models(models_to_keep=[predictor.model_best], dry_run=False)
    assert _files_under(predictor.path) == []
    assert len(predictor.predict(test)) == len(test)


def test_in_memory_predictor_keeps_the_cached_data_in_memory():
    df = _frame("binary")
    train = df.iloc[:150]
    predictor = TabularPredictor(
        label="y", path=tempfile.mkdtemp(prefix="ag_in_memory_"), verbosity=0, save_to_disk=False
    )
    predictor.fit(train, hyperparameters={DummyModel: {}}, validation_mode="none", fit_weighted_ensemble=False)
    trainer = predictor._trainer
    X = trainer.load_X()
    assert X is not None and len(X) == len(train)
    assert trainer.load_y() is not None
    assert _files_under(predictor.path) == []


def test_in_memory_bagged_fit_serves_out_of_fold_predictions():
    df = _frame("binary")
    train, test = df.iloc[:150], df.iloc[150:]
    predictor = TabularPredictor(
        label="y", path=tempfile.mkdtemp(prefix="ag_in_memory_"), verbosity=0, save_to_disk=False
    )
    predictor.fit(train, hyperparameters={DummyModel: {}}, num_bag_folds=2, fit_weighted_ensemble=False)
    assert _files_under(predictor.path) == []
    oof = predictor.predict_proba_oof()
    assert len(oof) == len(train)
    artifact = predictor.simulation_artifact(test_data=test)
    assert len(artifact["y_val"]) == len(train) and len(artifact["y_test"]) == len(test)
    assert len(artifact["pred_proba_dict_test"][predictor.model_best]) == len(test)


def test_default_predictor_still_saves_and_reloads():
    df = _frame("binary")
    train, test = df.iloc[:150], df.iloc[150:]
    path = tempfile.mkdtemp(prefix="ag_on_disk_")
    predictor = TabularPredictor(label="y", path=path, verbosity=0)
    predictor.fit(train, hyperparameters={DummyModel: {}}, validation_mode="none", fit_weighted_ensemble=False)
    assert len(_files_under(path)) > 0
    reloaded = TabularPredictor.load(path)
    np.testing.assert_array_equal(reloaded.predict(test).to_numpy(), predictor.predict(test).to_numpy())
