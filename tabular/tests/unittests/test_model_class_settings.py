"""`model_class_settings`: process-wide settings a model class owns, set once at fit and re-applied on load."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from autogluon.core.models.abstract._class_settings import ClassSettings
from autogluon.core.models.dummy.dummy_model import DummyModel
from autogluon.tabular import TabularPredictor
from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import TabPFNModel


@dataclass(frozen=True)
class _DummySettings(ClassSettings):
    knob: int = 1


class _SettingsDummyModel(DummyModel):
    class_settings_cls = _DummySettings


def _data(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    df["label"] = rng.integers(0, 2, n)
    return df


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch):
    monkeypatch.setattr(_SettingsDummyModel, "_class_settings", None, raising=False)
    monkeypatch.setattr(_SettingsDummyModel, "_class_settings_set", False, raising=False)


def test_fit_applies_model_class_settings_and_load_reapplies_them(tmp_path):
    predictor = TabularPredictor(label="label", path=str(tmp_path / "p"), verbosity=0).fit(
        _data(),
        hyperparameters={_SettingsDummyModel: {}},
        model_class_settings={_SettingsDummyModel: {"knob": 7}},
        fit_weighted_ensemble=False,
    )
    assert _SettingsDummyModel.get_class_settings().knob == 7
    assert predictor._learner.model_class_settings == {_SettingsDummyModel: {"knob": 7}}
    model = predictor._trainer.load_model(predictor.model_names()[0])
    assert model._class_settings_snapshot == {"knob": 7}

    _SettingsDummyModel._class_settings = None
    _SettingsDummyModel._class_settings_set = False
    loaded = TabularPredictor.load(predictor.path)
    assert _SettingsDummyModel.get_class_settings().knob == 7
    assert loaded.predict(_data(5)).shape == (5,)


def test_model_class_settings_resolve_model_keys_and_reject_unknown_settings(tmp_path, monkeypatch):
    predictor = TabularPredictor(label="label", path=str(tmp_path / "p"), verbosity=0)
    monkeypatch.setattr(TabPFNModel, "_class_settings", TabPFNModel.get_class_settings(), raising=False)
    predictor._apply_model_class_settings({"TABPFN-3": {"shared_network_capacity": 3}})
    assert TabPFNModel.get_class_settings().shared_network_capacity == 3, "TabPFN-3 shares the TabPFN base's settings"
    assert predictor._learner.model_class_settings == {"TABPFN-3": {"shared_network_capacity": 3}}

    with pytest.raises(ValueError, match="valid settings are"):
        TabularPredictor(label="label", path=str(tmp_path / "q"), verbosity=0).fit(
            _data(), hyperparameters={"DUMMY": {}}, model_class_settings={_SettingsDummyModel: {"knobb": 1}}
        )
    with pytest.raises(ValueError, match="declares no class settings"):
        TabularPredictor(label="label", path=str(tmp_path / "r"), verbosity=0).fit(
            _data(), hyperparameters={"DUMMY": {}}, model_class_settings={"DUMMY": {"anything": 1}}
        )
