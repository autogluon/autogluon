import tempfile

import pytest

from autogluon.forecasting.models.gluonts import (
    DeepARModel, AutoTabularModel, SimpleFeedForwardModel, MQCNNModel
)
from autogluon.forecasting.models.gluonts.abstract_gluonts import AbstractGluonTSModel

TESTABLE_MODELS = [
    # AutoTabularModel,  # TODO: enable tests when model is stabilized
    DeepARModel,
    MQCNNModel,
    SimpleFeedForwardModel,
]


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_models_initializable(model_class):
    with tempfile.TemporaryDirectory() as tp:
        model = model_class(tp, freq="H", prediction_length=24)
    assert isinstance(model, AbstractGluonTSModel)

