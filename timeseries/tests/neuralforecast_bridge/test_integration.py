"""Actual CPU NeuralForecast integration. NF_PYTHON selects the isolated backend."""
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("NF_PYTHON"), reason="Set NF_PYTHON to the pinned backend environment.")


def dataset(n_items=2):
    from autogluon.timeseries import TimeSeriesDataFrame

    frame = pd.concat([
        pd.DataFrame({"item_id": f"item-{index}", "timestamp": pd.date_range("2025-01-01", periods=64, freq="D"),
                      "target": 10 + index + np.sin(np.arange(64) / 3) + np.arange(64) * .03,
                      "x": np.cos(np.arange(64) / 3), "planned": np.arange(64) % 7})
        for index in range(n_items)
    ], ignore_index=True)
    return TimeSeriesDataFrame.from_data_frame(frame)


def parameters(**kwargs):
    return {"python_executable": os.environ["NF_PYTHON"], "input_size": 8,
            "max_steps": 2, "val_check_steps": 1, "windows_batch_size": 4,
            "batch_size": 2, "random_seed": 1, "num_lr_decays": -1, **kwargs}


def test_catalog_matches_actual_backend_and_native_aliases():
    from autogluon.timeseries.models import DeepARModel, ModelRegistry
    from autogluon.timeseries.models.neuralforecast import NEURALFORECAST_MODELS, NEURALFORECAST_REVISION

    code = "import json; from neuralforecast import models; print(json.dumps(models.__all__))"
    actual = json.loads(subprocess.check_output([os.environ["NF_PYTHON"], "-c", code], text=True).strip().splitlines()[-1])
    assert set(actual) == set(NEURALFORECAST_MODELS)
    for name in actual:
        cls = ModelRegistry.get_model_class(f"NF{name}")
        assert cls.nf_model_name == name
        assert cls(freq="D", path="/tmp/nf-registry-check").get_hyperparameters()["model_name"] == name
    assert ModelRegistry.get_model_class("DeepAR") is DeepARModel
    assert len(NEURALFORECAST_REVISION) == 40


@pytest.mark.parametrize("model_name,extra", [
    ("MLP", {"hidden_size": 8, "num_layers": 1, "loss": "MQLoss"}),
    ("NHITS", {"mlp_units": [[8, 8], [8, 8], [8, 8]], "loss": "MQLoss"}),
    ("DeepAR", {"lstm_hidden_size": 8, "lstm_n_layers": 1}),
    ("iTransformer", {"hidden_size": 8, "n_heads": 2, "e_layers": 1, "d_ff": 8, "loss": "MQLoss"}),
    ("MLP", {"hidden_size": 8, "num_layers": 1, "loss": "MSE"}),
])
def test_real_fit_predict_and_relocation(tmp_path, model_name, extra):
    from autogluon.timeseries.models import ModelRegistry
    from autogluon.timeseries.utils.features import CovariateMetadata

    cls = ModelRegistry.get_model_class(f"NF{model_name}")
    data = dataset()
    train = data.slice_by_timestep(None, -2)
    metadata = CovariateMetadata(past_covariates_real=["x"], known_covariates_real=["planned"])
    model = cls(path=str(tmp_path / "original"), freq="D", prediction_length=2,
                covariate_metadata=metadata, quantile_levels=[.1, .5, .9], eval_metric="MAE",
                hyperparameters=parameters(**extra))
    model.fit(train, time_limit=240, num_cpus=2)
    future = data.slice_by_timestep(-2, None)[["planned"]]
    forecast = model.predict(train, known_covariates=future)
    assert forecast.shape == (4, 4)
    assert np.isfinite(forecast.to_numpy()).all()
    assert forecast.index.equals(model.get_forecast_horizon_index(train))
    assert model.backend_info["training_end"]["0"] == str(train.loc["item-0"].index.max())
    if model_name == "MLP":
        assert model.backend_info["features"]["hist_exog_list"] == ["x"]
        assert model.backend_info["features"]["futr_exog_list"] == ["planned"]
    model.save()
    relocated = tmp_path / "relocated"
    shutil.copytree(model.path, relocated)
    shutil.rmtree(model.path)
    restored = cls.load(str(relocated))
    second = restored.predict(train, known_covariates=future)
    pd.testing.assert_frame_equal(forecast, second, check_exact=False, atol=1e-5, rtol=1e-5)


def test_hierarchical_reconciliation(tmp_path):
    from autogluon.timeseries.models.neuralforecast import NFHINTModel

    data = dataset(3)[["target"]]
    # Exact coherent hierarchy: item-0 = item-1 + item-2.
    data.loc[("item-0", slice(None)), "target"] = data.loc["item-1", "target"].to_numpy() + data.loc["item-2", "target"].to_numpy()
    model = NFHINTModel(path=str(tmp_path), freq="D", prediction_length=2, quantile_levels=[.1, .5, .9],
                        eval_metric="RMSE", hyperparameters=parameters(
                            base_model="MLP", hidden_size=8, num_layers=1,
                            hierarchy_item_ids=["item-0", "item-1", "item-2"],
                            S=[[1, 1], [1, 0], [0, 1]], reconciliation="BottomUp"))
    model.fit(data, time_limit=240, num_cpus=2)
    forecast = model.predict(data)
    np.testing.assert_allclose(forecast.loc["item-0", "mean"], forecast.loc["item-1", "mean"] + forecast.loc["item-2", "mean"], rtol=1e-5)
    model.save()
    restored = NFHINTModel.load(model.path)
    pd.testing.assert_frame_equal(forecast, restored.predict(data), check_exact=False, atol=1e-5, rtol=1e-5)


def test_predictor_backtests_ensemble_and_refit(tmp_path):
    from autogluon.timeseries import TimeSeriesPredictor

    data = dataset()
    predictor = TimeSeriesPredictor(path=str(tmp_path / "predictor"), prediction_length=2, freq="D", eval_metric="MAE",
                                    known_covariates_names=["planned"], quantile_levels=[.1, .5, .9])
    predictor.fit(data, hyperparameters={"NFMLP": parameters(hidden_size=8, num_layers=1, loss="MQLoss"), "Naive": {}},
                  num_val_windows=3, refit_every_n_windows=2, time_limit=420, enable_ensemble=True)
    board = predictor.leaderboard()
    assert board.model.str.contains("NFMLP").any(), board
    assert board.model.str.contains("Ensemble").any(), board
    assert np.isfinite(board.score_val).all()
    predictor.refit_full(model="all")
    future = predictor.make_future_data_frame(data)
    future["planned"] = 0.
    forecasts = predictor.predict(data, known_covariates=future)
    assert np.isfinite(forecasts.to_numpy()).all()


def test_process_timeout_and_bad_backend(tmp_path):
    from autogluon.core.utils.exceptions import TimeLimitExceeded
    from autogluon.timeseries.models.neuralforecast.model import _run_worker

    request = tmp_path / "request.json"
    request.write_text("{}")
    with pytest.raises(TimeLimitExceeded):
        _run_worker("fit", request, os.environ["NF_PYTHON"], .001)
    with pytest.raises(RuntimeError, match="Backend log"):
        _run_worker("fit", request, os.environ["NF_PYTHON"], 60)
