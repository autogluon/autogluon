import pytest

from autogluon.timeseries.models.ensemble.array_based.models import (
    LinearStackerEnsemble,
    MedianEnsemble,
    PerQuantileTabularEnsemble,
    TabularEnsemble,
)
from autogluon.timeseries.models.ensemble.array_based.regressor import (
    LinearStackerEnsembleRegressor,
    MedianEnsembleRegressor,
    PerQuantileTabularEnsembleRegressor,
    TabularEnsembleRegressor,
)

MODEL_TO_REGRESSOR = {
    MedianEnsemble: MedianEnsembleRegressor,
    TabularEnsemble: TabularEnsembleRegressor,
    PerQuantileTabularEnsemble: PerQuantileTabularEnsembleRegressor,
    LinearStackerEnsemble: LinearStackerEnsembleRegressor,
}


class TestEnsembleModels:
    @pytest.fixture(params=list(MODEL_TO_REGRESSOR.keys()))
    def model_class(self, request):
        yield request.param

    def test_given_model_when_initialized_then_ensemble_regressor_is_none(self, model_class):
        assert model_class().ensemble_regressor is None

    def test_given_model_when_fit_called_then_ensemble_regressor_created_and_fitted(self, model_class, ensemble_data):
        model = model_class()
        model.prediction_length = 5
        model.fit(**ensemble_data)

        assert model.ensemble_regressor is not None
        assert isinstance(model.ensemble_regressor, MODEL_TO_REGRESSOR[type(model)])

    def test_given_fitted_model_when_predict_called_then_prediction_returned(
        self, model_class, ensemble_data, ensemble_test_data
    ):
        model = model_class()
        model.prediction_length = 5
        model.fit(**ensemble_data)

        result = model._predict(ensemble_test_data)
        assert result is not None
        assert len(result) > 0
