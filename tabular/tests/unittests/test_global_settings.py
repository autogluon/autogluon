"""`global_settings`: process-wide AutoGluon settings set once at fit, re-applied on load and in worker processes."""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import autogluon.core.global_settings as gs
import autogluon.core.models.abstract._shared_weights_registry as registry
from autogluon.core.global_settings import GlobalSettings, get_global_settings
from autogluon.core.models.abstract._class_settings import ClassSettings
from autogluon.core.models.dummy.dummy_model import DummyModel
from autogluon.tabular import TabularPredictor


@dataclass(frozen=True)
class _DummySettings(ClassSettings):
    knob: int = 1


class _SettingsDummyModel(DummyModel):
    class_settings_cls = _DummySettings

    def _fit(self, **kwargs):
        self.settings_seen_in_fit = (registry.capacity(), type(self).get_class_settings().knob)
        return super()._fit(**kwargs)


def _data(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    df["label"] = rng.integers(0, 2, n)
    return df


def _start_fresh_process():
    gs._SETTINGS = GlobalSettings()
    gs._SET = False
    registry.set_capacity(None)
    _SettingsDummyModel._class_settings = None
    _SettingsDummyModel._class_settings_set = False


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch):
    monkeypatch.delenv(registry.CAPACITY_ENV_VAR, raising=False)
    monkeypatch.delenv(registry.LEGACY_CAPACITY_ENV_VAR, raising=False)
    monkeypatch.setattr(gs, "_SETTINGS", GlobalSettings())
    monkeypatch.setattr(gs, "_SET", False)
    monkeypatch.setattr(_SettingsDummyModel, "_class_settings", None, raising=False)
    monkeypatch.setattr(_SettingsDummyModel, "_class_settings_set", False, raising=False)
    registry.set_capacity(None)
    yield
    registry.set_capacity(None)


def test_fit_applies_global_settings_and_load_reapplies_them(tmp_path):
    predictor = TabularPredictor(label="label", path=str(tmp_path / "p"), verbosity=0).fit(
        _data(),
        hyperparameters={_SettingsDummyModel: {}},
        global_settings={"shared_weights_capacity": 5},
        fit_weighted_ensemble=False,
    )
    assert registry.capacity() == 5
    assert predictor._learner.global_settings == {"shared_weights_capacity": 5}
    model = predictor._trainer.load_model(predictor.model_names()[0])
    assert model._global_settings_snapshot == {"shared_weights_capacity": 5}
    assert model.settings_seen_in_fit[0] == 5

    _start_fresh_process()
    loaded = TabularPredictor.load(predictor.path)
    assert get_global_settings().shared_weights_capacity == 5
    assert registry.capacity() == 5
    assert loaded.predict(_data(5)).shape == (5,)


def test_uninitialized_model_fit_in_a_fresh_process_runs_under_the_launching_settings():
    """The fold template a bag ships to its parallel workers: pickled before `initialize`, fit where settings are defaults."""
    gs.set_global_settings(shared_weights_capacity=4)
    _SettingsDummyModel.set_class_settings(knob=7)
    payload = pickle.dumps(_SettingsDummyModel(name="fold", problem_type="binary", eval_metric="log_loss"))

    _start_fresh_process()
    fold_model = pickle.loads(payload)
    data = _data()
    fold_model.fit(X=data.drop(columns="label"), y=data["label"])
    assert fold_model.settings_seen_in_fit == (4, 7)
    assert fold_model._global_settings_snapshot == {"shared_weights_capacity": 4}
    assert fold_model._class_settings_snapshot == {"knob": 7}


def test_global_settings_reject_unknown_keys_before_training(tmp_path):
    with pytest.raises(ValueError, match="valid settings are"):
        TabularPredictor(label="label", path=str(tmp_path / "p"), verbosity=0).fit(
            _data(), hyperparameters={"DUMMY": {}}, global_settings={"shared_weight_capacity": 1}
        )
    with pytest.raises(ValueError, match="non-negative int or None"):
        TabularPredictor(label="label", path=str(tmp_path / "q"), verbosity=0).fit(
            _data(), hyperparameters={"DUMMY": {}}, global_settings={"shared_weights_capacity": -1}
        )
