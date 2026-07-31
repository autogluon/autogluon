"""Tests for exporting Chronos models as standalone checkpoints via `export_model`."""

import os

import numpy as np
import pytest

from autogluon.timeseries import TimeSeriesPredictor
from autogluon.timeseries.models import Chronos2Model, ChronosModel

from ...common import DUMMY_TS_DATAFRAME
from ..common import CHRONOS2_MODEL_PATH, CHRONOS_BOLT_MODEL_PATH, CHRONOS_CLASSIC_MODEL_PATH

PREDICTION_LENGTH = 3


def load_exported_pipeline(path):
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(path, device_map="cpu")


def get_inner_model(model):
    """The underlying `transformers` model wrapped by an AutoGluon Chronos model."""
    if isinstance(model, Chronos2Model):
        model.load_model_pipeline()
        return model._model_pipeline.inner_model
    return model.model_pipeline.inner_model


# The original Chronos models generate forecasts by sampling, so their predictions are not reproducible across
# `predict` calls. For these models we compare the exported weights instead of the predictions.
DETERMINISTIC_MODELS = [
    (ChronosModel, CHRONOS_BOLT_MODEL_PATH),
    (Chronos2Model, CHRONOS2_MODEL_PATH),
]
ALL_MODELS = DETERMINISTIC_MODELS + [(ChronosModel, CHRONOS_CLASSIC_MODEL_PATH)]


@pytest.mark.parametrize("model_class, model_path", DETERMINISTIC_MODELS)
def test_when_zero_shot_model_exported_then_predictions_are_unchanged(model_class, model_path, tmp_path):
    model = model_class(
        path=str(tmp_path / "model"),
        prediction_length=PREDICTION_LENGTH,
        hyperparameters={"model_path": model_path, "device": "cpu"},
    )
    model.fit(DUMMY_TS_DATAFRAME)
    expected = model.predict(DUMMY_TS_DATAFRAME)

    export_path = model.export_model(tmp_path / "export")

    exported_model = model_class(
        path=str(tmp_path / "reimported"),
        prediction_length=PREDICTION_LENGTH,
        hyperparameters={"model_path": export_path, "device": "cpu"},
    )
    exported_model.fit(DUMMY_TS_DATAFRAME)
    predictions = exported_model.predict(DUMMY_TS_DATAFRAME)

    assert np.allclose(expected.values, predictions.values, rtol=1e-4)


@pytest.mark.parametrize("model_class, model_path", ALL_MODELS)
def test_when_zero_shot_model_exported_then_weights_are_unchanged(model_class, model_path, tmp_path):
    model = model_class(
        path=str(tmp_path / "model"),
        prediction_length=PREDICTION_LENGTH,
        hyperparameters={"model_path": model_path, "device": "cpu"},
    )
    model.fit(DUMMY_TS_DATAFRAME)
    expected_state_dict = get_inner_model(model).state_dict()

    export_path = model.export_model(tmp_path / "export")

    exported_state_dict = load_exported_pipeline(export_path).inner_model.state_dict()

    assert exported_state_dict.keys() == expected_state_dict.keys()
    for key, expected_weight in expected_state_dict.items():
        assert (exported_state_dict[key] == expected_weight).all(), f"Weight mismatch for {key}"


@pytest.mark.parametrize(
    "model_class, model_path",
    [
        (ChronosModel, CHRONOS_BOLT_MODEL_PATH),
        (Chronos2Model, CHRONOS2_MODEL_PATH),
    ],
)
def test_when_model_exported_then_checkpoint_can_be_loaded_by_chronos(model_class, model_path, tmp_path):
    model = model_class(
        path=str(tmp_path / "model"),
        prediction_length=PREDICTION_LENGTH,
        hyperparameters={"model_path": model_path, "device": "cpu"},
    )
    model.fit(DUMMY_TS_DATAFRAME)

    export_path = model.export_model(tmp_path / "export")

    assert os.path.isfile(os.path.join(export_path, "config.json"))
    assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
    assert load_exported_pipeline(export_path) is not None


@pytest.mark.parametrize(
    "model_class, model_path",
    [
        (ChronosModel, CHRONOS_BOLT_MODEL_PATH),
        (Chronos2Model, CHRONOS2_MODEL_PATH),
    ],
)
def test_when_model_exported_then_export_contains_no_symlinks(model_class, model_path, tmp_path):
    model = model_class(
        path=str(tmp_path / "model"),
        prediction_length=PREDICTION_LENGTH,
        hyperparameters={"model_path": model_path, "device": "cpu"},
    )
    model.fit(DUMMY_TS_DATAFRAME)

    export_path = model.export_model(tmp_path / "export")

    symlinks = [
        os.path.join(root, name)
        for root, dirs, files in os.walk(export_path)
        for name in dirs + files
        if os.path.islink(os.path.join(root, name))
    ]
    assert symlinks == []


@pytest.mark.parametrize(
    "transform, value",
    [
        ("target_scaler", "standard"),
        ("covariate_scaler", "global"),
        ("covariate_regressor", "LR"),
    ],
)
def test_when_model_uses_transforms_then_export_raises(transform, value, df_with_covariates, tmp_path):
    data, covariate_metadata = df_with_covariates
    model = ChronosModel(
        path=str(tmp_path / "model"),
        prediction_length=PREDICTION_LENGTH,
        covariate_metadata=covariate_metadata,
        hyperparameters={"model_path": CHRONOS_BOLT_MODEL_PATH, "device": "cpu", transform: value},
    )
    model.fit(data)
    assert getattr(model, transform) is not None, f"{transform} was not initialized, the test is not meaningful"

    with pytest.raises(ValueError, match=f"cannot be exported.*{transform}"):
        model.export_model(tmp_path / "export")


class TestExportFineTunedChronos2:
    @pytest.fixture(scope="class")
    def fine_tuned_predictor(self, tmp_path_factory):
        predictor = TimeSeriesPredictor(
            prediction_length=PREDICTION_LENGTH,
            path=str(tmp_path_factory.mktemp("fine_tuned_predictor")),
            verbosity=0,
        )
        predictor.fit(
            DUMMY_TS_DATAFRAME,
            hyperparameters={
                "Chronos2": {
                    "model_path": CHRONOS2_MODEL_PATH,
                    "device": "cpu",
                    "fine_tune": True,
                    "fine_tune_steps": 2,
                    "fine_tune_batch_size": 4,
                }
            },
            skip_model_selection=True,
        )
        return predictor

    def test_when_fine_tuned_model_exported_then_lora_adapter_is_merged(self, fine_tuned_predictor, tmp_path):
        export_path = fine_tuned_predictor.export_model(tmp_path / "export")

        # a merged checkpoint contains full model weights instead of adapter weights
        assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
        assert not os.path.exists(os.path.join(export_path, "adapter_config.json"))
        assert load_exported_pipeline(export_path) is not None

    def test_when_fine_tuned_model_exported_then_predictions_are_unchanged(self, fine_tuned_predictor, tmp_path):
        expected = fine_tuned_predictor.predict(DUMMY_TS_DATAFRAME)

        export_path = fine_tuned_predictor.export_model(tmp_path / "export")

        reimported_predictor = TimeSeriesPredictor(
            prediction_length=PREDICTION_LENGTH, path=str(tmp_path / "reimported"), verbosity=0
        )
        reimported_predictor.fit(
            DUMMY_TS_DATAFRAME,
            hyperparameters={"Chronos2": {"model_path": export_path, "device": "cpu"}},
            skip_model_selection=True,
        )
        predictions = reimported_predictor.predict(DUMMY_TS_DATAFRAME)

        assert np.allclose(expected.values, predictions.values, rtol=1e-4)

    def test_when_predictor_loaded_from_disk_then_model_can_be_exported(self, fine_tuned_predictor, tmp_path):
        loaded_predictor = TimeSeriesPredictor.load(fine_tuned_predictor.path)

        export_path = loaded_predictor.export_model(tmp_path / "export")

        assert load_exported_pipeline(export_path) is not None
