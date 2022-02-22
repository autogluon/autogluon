"""Unit tests and utils common to all models"""
import pytest

from .common import DUMMY_DATASET, dict_equal_primitive

from .test_gluonts import TESTABLE_MODELS as GLUONTS_TESTABLE_MODELS


TESTABLE_MODELS = GLUONTS_TESTABLE_MODELS


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_models_saved_then_they_can_be_loaded(model_class, temp_model_path):
    model = model_class(
        path=temp_model_path,
        freq="H",
        hyperparameters={
            "epochs": 1,
            "ag_args_fit": {"quantile_levels": [0.1, 0.9]},
        },
    )
    model.fit(
        train_data=DUMMY_DATASET,
    )
    model.save()

    loaded_model = model_class.load(path=model.path)

    assert dict_equal_primitive(model.params, loaded_model.params)
    assert dict_equal_primitive(model.params_aux, loaded_model.params_aux)
