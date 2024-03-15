import pytest

from autogluon.timeseries.models import ChronosModel

from ..common import DATAFRAME_WITH_COVARIATES, DATAFRAME_WITH_STATIC, DUMMY_TS_DATAFRAME

DATASETS = [DUMMY_TS_DATAFRAME, DATAFRAME_WITH_STATIC, DATAFRAME_WITH_COVARIATES]
TESTABLE_MODELS = [ChronosModel]
HYPERPARAMETER_SETS = []


@pytest.fixture(scope="module")
def default_chronos_tiny_model() -> ChronosModel:
    model = ChronosModel(
        hyperparameters={
            "model_path": "amazon/chronos-t5-tiny",
            "device": "cpu",
        },
    )
    model.fit(train_data=None)
    return model


@pytest.mark.parametrize("data", DATASETS)
def test_when_on_cpu_then_chronos_model_can_score_and_cache_oof(data, default_chronos_tiny_model):
    default_chronos_tiny_model.score_and_cache_oof(data)
    if not default_chronos_tiny_model.skip_validation:
        assert default_chronos_tiny_model._oof_predictions is not None


@pytest.mark.parametrize("data", DATASETS)
def test_when_on_cpu_then_chronos_model_can_infer(data, default_chronos_tiny_model):
    predictions = default_chronos_tiny_model.predict(data)
    assert predictions is not None
