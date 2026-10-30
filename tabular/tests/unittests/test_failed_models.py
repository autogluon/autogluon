"""A model whose fit fails: the trainer skips it and keeps nothing it held."""

from __future__ import annotations

import gc
import weakref

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
