"""`TabularPredictor.persist` runs every persisted model's untimed `prepare_for_inference`, bagged children included."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from autogluon.core.models.dummy.dummy_model import DummyModel
from autogluon.tabular import TabularPredictor

#: ``(class name, model name)`` of every ``prepare_for_inference`` call; bagged children are named by their fold.
CALLS: list[tuple[str, str]] = []


class _RecordingModel(DummyModel):
    ag_key = "RECORDING_DUMMY"
    ag_name = "RecordingDummy"

    def prepare_for_inference(self) -> None:
        CALLS.append((type(self).__name__, self.name))


class _FailingModel(_RecordingModel):
    ag_key = "FAILING_DUMMY"
    ag_name = "FailingDummy"

    def prepare_for_inference(self) -> None:
        super().prepare_for_inference()
        raise RuntimeError("cannot prepare")


def _data(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    df["label"] = rng.integers(0, 2, n)
    return df


def test_persist_prepares_every_model_object_once_including_bagged_children(tmp_path, caplog):
    CALLS.clear()
    predictor = TabularPredictor(label="label", path=str(tmp_path / "p"), verbosity=0).fit(
        _data(),
        hyperparameters={_RecordingModel: {}, _FailingModel: {}},
        num_bag_folds=2,
        fit_weighted_ensemble=False,
    )
    assert CALLS == [], "fit never calls the untimed hook"
    names = predictor.model_names()
    assert len(names) == 2

    # `verbosity=0` quiets the `autogluon` logger tree, so listen on the bag's own logger.
    with caplog.at_level(logging.WARNING, logger="autogluon.core.models.ensemble.bagged_ensemble_model"):
        persisted = predictor.persist(models="all")
    assert sorted(persisted) == sorted(names)
    recording_children = [name for cls, name in CALLS if cls == "_RecordingModel"]
    failing_children = [name for cls, name in CALLS if cls == "_FailingModel"]
    assert sorted(recording_children) == ["S1F1", "S1F2"], "one call per bagged child"
    assert sorted(failing_children) == ["S1F1", "S1F2"], "a failing child does not stop the others"
    assert any("prepare_for_inference failed for" in record.message for record in caplog.records)
    assert predictor.model_names(persisted=True) and predictor.predict(_data(5)).shape == (5,)

    # Already persisted models are not prepared again; unpersisting and persisting runs the hook once more.
    n_calls = len(CALLS)
    predictor.persist(models="all")
    assert len(CALLS) == n_calls
    predictor.unpersist()
    predictor.persist(models=names[:1], max_memory=None)
    assert len(CALLS) == n_calls + 2
