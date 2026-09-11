"""Dependency-light boundary checks; these do not claim checkpoint/model accuracy."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2] / "src/autogluon/timeseries/models/neuralforecast"

def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

worker = load_module("_worker")
catalog = load_module("_catalog")

class Array:
    def __init__(self, values):
        self.values = np.asarray(values)
    def detach(self):
        return self
    def cpu(self):
        return self.values

class Model:
    def __init__(self, loss):
        self.loss = loss
    def __repr__(self):
        return "forecast"

@pytest.mark.parametrize("name", catalog.NEURALFORECAST_MODELS)
def test_catalog_names(name):
    assert name.isidentifier()
    assert catalog.NEURALFORECAST_MODELS.count(name) == 1
    assert len(catalog.NEURALFORECAST_MODELS) == 66
    assert len(catalog.NEURALFORECAST_REVISION) == 40

@pytest.mark.parametrize("supported", [False, True])
def test_covariate_roles(supported):
    cls = type("Example", (), {"EXOGENOUS_HIST": supported, "EXOGENOUS_FUTR": True, "EXOGENOUS_STAT": False})
    config = {}
    selected, ignored = worker.select_features(cls, config, {"past": ["vix"], "known": ["holiday"], "static": ["country_id"]})
    assert selected["futr_exog_list"] == ["holiday"]
    assert selected["hist_exog_list"] == (["vix"] if supported else [])
    assert "country_id" in ignored
    assert "vix" not in selected["futr_exog_list"]

@pytest.mark.parametrize("values", [["vix"], ["holiday", "holiday"], "holiday", [3]])
def test_future_cannot_receive_unavailable_or_invalid_columns(values):
    cls = type("Example", (), {"EXOGENOUS_FUTR": True})
    with pytest.raises(ValueError):
        worker.select_features(cls, {"futr_exog_list": values}, {"past": ["vix"], "known": ["holiday"], "static": []})

def test_explicit_unsupported_covariate_rejected():
    with pytest.raises(ValueError, match="does not support"):
        worker.select_features(type("Example", (), {}), {"hist_exog_list": ["vix"]}, {"past": ["vix"], "known": [], "static": []})

@pytest.mark.parametrize("native", [False, True])
def test_quantile_labels(native):
    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    raw = pd.DataFrame({"unique_id": [0, 0], "ds": dates})
    qs = [0.1, 0.5, 0.9]
    loss = SimpleNamespace()
    if native:
        loss.quantiles = Array(np.asarray(qs, dtype=np.float32))
        loss.output_names = ["-lo-80", "-median", "-hi-80"]
        for i, suffix in enumerate(loss.output_names):
            raw["forecast" + suffix] = i + np.arange(2)
    else:
        for i, q in enumerate(qs):
            raw[f"forecast_ql{q}"] = i + np.arange(2)
        raw["forecast"] = [1.25, 2.25]
    output = worker.normalize_forecasts(raw, Model(loss), qs)
    assert output["0.1"].tolist() == [0, 1]
    assert output["0.9"].tolist() == [2, 3]
    assert output["mean"].tolist() == ([1, 2] if native else [1.25, 2.25])
    assert output["ds"].tolist() == dates.astype("int64").tolist()
    with pytest.raises(ValueError, match="did not return quantile"):
        worker.normalize_forecasts(raw, Model(loss), [0.05, 0.5])

def test_nonfinite_output_rejected():
    raw = pd.DataFrame({"unique_id": [0], "ds": [pd.Timestamp("2026-01-01")], "forecast": [np.nan], "forecast_ql0.5": [1.]})
    with pytest.raises(ValueError, match="non-finite"):
        worker.normalize_forecasts(raw, Model(SimpleNamespace()), [0.5])

def test_synchronized_panel_and_nanosecond_roundtrip(tmp_path):
    dates = pd.date_range("2026-01-01 00:00:00.000000007", periods=2, freq="D")
    frame = pd.DataFrame({"unique_id": [0, 0, 1, 1], "ds": list(dates) * 2, "y": [1., 2., 3., 4.]})
    worker.check_panel(frame, [0, 1])
    serialized = frame.assign(ds=frame.ds.astype("int64"))
    serialized.to_csv(tmp_path / "history.csv", index=False)
    pd.testing.assert_frame_equal(worker.read_frame(tmp_path / "history.csv"), frame)
    with pytest.raises(ValueError, match="same complete item panel"):
        worker.check_panel(frame, [0])
    with pytest.raises(ValueError, match="synchronized"):
        worker.check_panel(frame.iloc[:-1], [0, 1])
    with pytest.raises(ValueError, match="empty"):
        worker.check_panel(frame.iloc[:0], [])
