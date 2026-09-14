import os


def test_best_quality_all_nf_contains_default_and_all_neuralforecast_models(monkeypatch, tmp_path):
    backend = tmp_path / "nf-python"
    backend.touch()
    monkeypatch.setenv("AUTOGLUON_NF_PYTHON", os.fspath(backend))

    from autogluon.timeseries.configs import get_hyperparameter_presets, get_predictor_presets
    from autogluon.timeseries.models.neuralforecast import NEURALFORECAST_MODELS
    from autogluon.timeseries.trainer.model_set_builder import HyperparameterBuilder

    hyperparameter_presets = get_hyperparameter_presets()
    predictor_presets = get_predictor_presets()

    all_nf = hyperparameter_presets["default_all_nf"]
    default = hyperparameter_presets["default"]

    assert set(default).issubset(all_nf)
    assert {f"NF{name}" for name in NEURALFORECAST_MODELS}.issubset(all_nf)
    assert len({name for name in all_nf if name.startswith("NF")}) == len(NEURALFORECAST_MODELS) == 66
    assert all_nf["Chronos2"] == default["Chronos2"]
    assert all(all_nf[f"NF{name}"]["python_executable"] == os.fspath(backend) for name in NEURALFORECAST_MODELS)

    resolved = HyperparameterBuilder(
        hyperparameters="default_all_nf",
        hyperparameter_tune=False,
        excluded_model_types=None,
    ).get_hyperparameters()
    assert set(default).issubset(resolved)
    assert {f"NF{name}" for name in NEURALFORECAST_MODELS}.issubset(resolved)

    assert predictor_presets["best_quality_all_nf"] == {
        "hyperparameters": "default_all_nf",
        "num_val_windows": "auto",
        "refit_every_n_windows": "auto",
    }
