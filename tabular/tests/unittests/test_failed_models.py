"""A model whose fit fails: the trainer skips it and keeps nothing it held or wrote."""

from __future__ import annotations

import gc
import os
import re
import weakref
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autogluon.core.models import DummyModel
from autogluon.tabular import TabularPredictor


class FailingDummyModel(DummyModel):
    """A dummy model whose fit holds an array, then raises; `held` keeps a weak reference to each array."""

    ag_name = "FailingDummy"
    held: list[weakref.ref] = []

    def _fit(self, X, y, **kwargs):
        array = np.ones(1_000)
        FailingDummyModel.held.append(weakref.ref(array))
        raise RuntimeError("this fit fails")


class WritingDummyModel(DummyModel):
    """A dummy model whose fit writes a file into its directory, then raises from fold `fail_from` on."""

    ag_name = "WritingDummy"

    def _fit(self, X, y, **kwargs):
        os.makedirs(self.path, exist_ok=True)
        with open(os.path.join(self.path, "checkpoint"), "wb") as f:
            f.write(b"0" * 1_000)
        fold = re.search(r"F(\d+)$", self.name)
        if (int(fold.group(1)) if fold else 1) >= self._get_model_params()["fail_from"]:
            raise RuntimeError("this fit fails")
        return super()._fit(X, y, **kwargs)


def _train(rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(rows, 3)), columns=["a", "b", "c"])
    df["label"] = (df["a"] > 0).astype(int)
    return df


@pytest.mark.parametrize("num_bag_folds", [0, 2], ids=["not_bagged", "bagged"])
def test_a_failed_fit_frees_what_it_held_without_a_garbage_collection(num_bag_folds, tmp_path):
    FailingDummyModel.held.clear()
    gc.disable()
    try:
        TabularPredictor(label="label", path=str(tmp_path / "predictor"), verbosity=0).fit(
            _train(), hyperparameters={FailingDummyModel: {}, "DUMMY": {}}, num_bag_folds=num_bag_folds, num_gpus=0
        )
        assert FailingDummyModel.held, "the failing model was not fit"
        assert all(ref() is None for ref in FailingDummyModel.held)
    finally:
        gc.enable()


@pytest.mark.parametrize(
    ("num_bag_folds", "fail_from"), [(0, 1), (2, 2)], ids=["not_bagged", "bag_fails_in_its_second_fold"]
)
def test_a_failed_model_leaves_nothing_on_disk(num_bag_folds, fail_from, tmp_path):
    predictor = TabularPredictor(label="label", path=str(tmp_path / "predictor"), verbosity=0).fit(
        _train(),
        hyperparameters={WritingDummyModel: {"fail_from": fail_from}, "DUMMY": {}},
        num_bag_folds=num_bag_folds,
        num_gpus=0,
    )
    trainer = predictor._trainer
    (failed,) = trainer._models_failed_to_train_errors
    assert failed.startswith("WritingDummy")
    assert not os.path.exists(os.path.join(trainer.path, failed))


def test_a_bag_whose_remaining_folds_fail_keeps_only_its_first_folds_on_disk(tmp_path):
    predictor = TabularPredictor(label="label", path=str(tmp_path / "predictor"), verbosity=0).fit(
        _train(400),
        hyperparameters={WritingDummyModel: {"fail_from": 2}},
        num_bag_folds=4,
        num_bag_sets=1,
        validation_folds=1,
        dynamic_stacking=False,
        num_gpus=0,
    )
    trainer = predictor._trainer
    (bag,) = [m for m in predictor.model_names() if m.startswith("WritingDummy")]
    assert bag in trainer._models_failed_to_train_errors  # completing its remaining folds failed
    model = trainer.load_model(bag)
    assert [child if isinstance(child, str) else child.name for child in model.models] == ["S1F1"]
    assert sorted(p.name for p in Path(model.path).iterdir() if p.is_dir() and p.name != "utils") == ["S1F1"]
    assert np.isfinite(predictor.predict_proba(_train(50)).to_numpy()).all()
