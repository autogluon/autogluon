import itertools
from typing import Optional, Tuple
from unittest import mock

import pandas as pd
import pytest

from autogluon.core.constants import AG_ARGS_FIT
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.features import CovariateMetadata

from ..common import get_data_frame_with_item_index


class ConcreteTimeSeriesModel(AbstractTimeSeriesModel):
    """A dummy model that predicts 42s, implemented according to the custom model
    tutorial [1].
    
    References
    ----------
    .. [1] https://auto.gluon.ai/dev/tutorials/timeseries/advanced/forecasting-custom-model.html#implement-the-custom-model
    """
    _supports_known_covariates: bool = True
    _supports_past_covariates: bool = True
    _supports_static_features: bool = True

    def preprocess(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        is_train: bool = False,
        **kwargs,
    ) -> Tuple[TimeSeriesDataFrame, Optional[TimeSeriesDataFrame]]:
        """Method that implements model-specific preprocessing logic.

        This method is called on all data that is passed to `_fit` and `_predict` methods.
        """
        data = data.fill_missing_values(method="constant", value=42.0)
        return data, known_covariates

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[float] = None,
        **kwargs,
    ) -> None:
        # _fit depends on _get_model_params() to provide parameters for the inner model
        _ = self._get_model_params()

        # let's do some work
        self.dummy_learned_parameters = train_data.groupby(level=0).mean().to_dict()

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        """Predict future target given the historic time series data and the future values of known_covariates."""
        assert self.dummy_learned_parameters is not None

        return TimeSeriesDataFrame(
            pd.DataFrame(
                index=self.get_forecast_horizon_index(data), 
                columns=["mean"] + self.quantile_levels
            ).fillna(42.0)
        )


class TestAbstractModelFunctionality:
    
    @pytest.fixture(
        scope="class", 
        params=list(
            itertools.product(
                [["A", "B"], ["C", "D"], ["A"], [0, 1, 2, 3]],
                [10, 45],
                ["h", "T", "d"],
            )
        ),
    )
    def train_data(self, request):
        index, data_length, freq = request.param
        return get_data_frame_with_item_index(index, data_length=data_length, freq=freq)
    
    @pytest.fixture()
    def model(self, temp_model_path) -> ConcreteTimeSeriesModel:
        return ConcreteTimeSeriesModel(path=temp_model_path)
    
    def test_when_model_is_initialized_then_key_fields_set_correctly(self, temp_model_path):
        model = ConcreteTimeSeriesModel(
            path=temp_model_path,
            freq="h",
            prediction_length=3,
            quantile_levels=[0.1, 0.9],
            eval_metric="MAPE",
            hyperparameters={"some": "params"},
        )
        
        assert model.freq == "h"
        assert model.prediction_length == 3
        assert tuple(model.quantile_levels) == (0.1, 0.5, 0.9)
        assert model.eval_metric.name == "MAPE"
        
    def test_when_model_receives_median_then_must_not_drop_median_set_to_false(self, temp_model_path):
        model = ConcreteTimeSeriesModel(
            path=temp_model_path,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        assert not model.must_drop_median

    def test_when_model_does_not_receive_median_then_must_not_drop_median_set_to_true(self, temp_model_path):
        model = ConcreteTimeSeriesModel(
            path=temp_model_path,
            quantile_levels=[0.1, 0.9],
        )
        assert not model.must_drop_median
    
    def test_when_model_saved_and_loaded_with_load_oof_then_load_oof_called(self, model):        
        model.save()
        with mock.patch.object(model.__class__, "load_oof_predictions") as mock_load_oof:
            model.__class__.load(model.path, load_oof=True)
            mock_load_oof.assert_called_once()

    def test_when_support_model_covariate_properties_are_accessed_then_their_values_are_correct(self, model):
        assert model.supports_known_covariates == model.__class__._supports_known_covariates
        assert model.supports_past_covariates == model.__class__._supports_past_covariates
        assert model.supports_static_features == model.__class__._supports_static_features
        
    def test_when_model_is_initialized_with_ag_args_fit_then_they_are_included_in_get_params(self, train_data, temp_model_path):
        model = ConcreteTimeSeriesModel(
            path=temp_model_path,
            hyperparameters={AG_ARGS_FIT: {"key": "value"}},  # type: ignore
        )
        model.fit(train_data=train_data)
        assert AG_ARGS_FIT in model.get_params()["hyperparameters"]
        
    def test_when_model_is_fit_with_time_limit_less_than_zero_then_error_raised(self, train_data, model):
        with pytest.raises(TimeLimitExceeded):
            model.fit(train_data=train_data, time_limit=-1)

    @pytest.mark.parametrize(
        "covariate_regressor_hyperparameter", [
            "dummy_argument",
            {"key": "value"},
        ]
    )
    def test_when_create_covariate_regressor_is_called_then_covariate_regressor_is_constructed(
        self, 
        temp_model_path,
        covariate_regressor_hyperparameter,
    ):
        model = ConcreteTimeSeriesModel(
            path=temp_model_path,
            hyperparameters={"covariate_regressor": covariate_regressor_hyperparameter},
            metadata=CovariateMetadata(
                known_covariates_real=["dummy_column"]
            )
        )
        
        with mock.patch("autogluon.timeseries.models.abstract.abstract_timeseries_model.CovariateRegressor") as mock_covariate_regressor:
            model.initialize()  # calls create_covariate_regressor
            mock_covariate_regressor.assert_called_once()
            assert mock_covariate_regressor.call_args.kwargs["target"] == model.target
            assert mock_covariate_regressor.call_args.kwargs["metadata"] is model.metadata
            if isinstance(covariate_regressor_hyperparameter, dict):
                assert mock_covariate_regressor.call_args.kwargs["key"] == "value"
            else:
                assert mock_covariate_regressor.call_args.args[0] == "dummy_argument"

    def test_when_get_memory_size_called_then_memory_size_is_none(self, model):
        assert model.get_memory_size() is None
        
    def test_when_hyperparameter_tune_called_with_empty_search_space_then_skip_hpo_called(
        self, model: ConcreteTimeSeriesModel, train_data: TimeSeriesDataFrame
    ):
        with mock.patch("autogluon.timeseries.models.abstract.abstract_timeseries_model.skip_hpo") as mock_skip_hpo:
            model.hyperparameter_tune(
                hyperparameter_tune_kwargs="auto", 
                hpo_executor=None,
                train_data=train_data,
                val_data=train_data,
            )
            
            assert mock_skip_hpo.called
    
    # def test_when_time_limit_is_capped_then