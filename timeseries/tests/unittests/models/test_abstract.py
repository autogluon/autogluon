from unittest import mock

import pandas as pd
import pytest

from autogluon.core.constants import AG_ARGS_FIT, REFIT_FULL_SUFFIX
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

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: TimeSeriesDataFrame | None = None,
        time_limit: float | None = None,
        **kwargs,
    ) -> None:
        # _fit depends on get_hyperparameters() to provide parameters for the inner model
        _ = self.get_hyperparameters()

        # let's do some work
        self.dummy_learned_parameters = train_data.groupby(level=0).mean().to_dict()

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame | None = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        """Predict future target given the historical time series data and the future values of known_covariates."""
        assert self.dummy_learned_parameters is not None

        return TimeSeriesDataFrame(
            pd.DataFrame(
                index=self.get_forecast_horizon_index(data),
                columns=[str(q) for q in self.quantile_levels] + ["mean"],
            ).fillna(42.0)
        )


@pytest.fixture(scope="module")
def train_data():
    return get_data_frame_with_item_index(["A", "B"], data_length=100, freq="h")


def test_when_model_is_initialized_then_key_fields_set_correctly(temp_model_path):
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


def test_when_model_receives_median_then_must_not_drop_median_set_to_false(temp_model_path):
    model = ConcreteTimeSeriesModel(
        path=temp_model_path,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    assert not model.must_drop_median


def test_when_model_does_not_receive_median_then_must_not_drop_median_set_to_true(temp_model_path):
    model = ConcreteTimeSeriesModel(
        path=temp_model_path,
        quantile_levels=[0.1, 0.9],
    )
    assert model.must_drop_median


def test_when_model_saved_and_loaded_with_load_oof_then_load_oof_called(temp_model_path):
    model = ConcreteTimeSeriesModel(path=temp_model_path)
    model.save()
    with mock.patch.object(model.__class__, "load_oof_predictions") as mock_load_oof:
        model.__class__.load(model.path, load_oof=True)
        mock_load_oof.assert_called_once()


def test_when_support_model_covariate_properties_are_accessed_then_their_values_are_correct(temp_model_path):
    model = ConcreteTimeSeriesModel(path=temp_model_path)

    assert model.supports_known_covariates == model.__class__._supports_known_covariates
    assert model.supports_past_covariates == model.__class__._supports_past_covariates
    assert model.supports_static_features == model.__class__._supports_static_features


def test_when_model_is_initialized_with_ag_args_fit_then_they_are_included_in_get_params(train_data, temp_model_path):
    model = ConcreteTimeSeriesModel(
        path=temp_model_path,
        hyperparameters={AG_ARGS_FIT: {"key": "value"}},  # type: ignore
    )
    model.fit(train_data=train_data)
    assert AG_ARGS_FIT in model.get_params()["hyperparameters"]


@pytest.mark.parametrize(
    "covariate_regressor_hyperparameter",
    [
        "dummy_argument",
        {"key": "value"},
    ],
)
def test_when_create_covariate_regressor_is_called_then_covariate_regressor_is_constructed(
    temp_model_path,
    covariate_regressor_hyperparameter,
    train_data,
):
    model = ConcreteTimeSeriesModel(
        path=temp_model_path,
        hyperparameters={"covariate_regressor": covariate_regressor_hyperparameter},
        covariate_metadata=CovariateMetadata(known_covariates_real=["dummy_column"]),
    )
    with mock.patch(
        "autogluon.timeseries.models.abstract.abstract_timeseries_model.get_covariate_regressor"
    ) as mock_get_covariate_regressor:
        model.fit(train_data=train_data)

        mock_get_covariate_regressor.assert_called_once()
        assert mock_get_covariate_regressor.call_args.kwargs["target"] == model.target
        assert mock_get_covariate_regressor.call_args.kwargs["covariate_metadata"] is model.covariate_metadata
        assert type(mock_get_covariate_regressor.call_args.args[0]) is type(covariate_regressor_hyperparameter)


def test_when_hyperparameter_tune_called_with_empty_search_space_then_skip_hpo_called(temp_model_path, train_data):
    model = ConcreteTimeSeriesModel(path=temp_model_path)
    with mock.patch("autogluon.timeseries.models.abstract.tunable.skip_hpo") as mock_skip_hpo:
        model.hyperparameter_tune(
            hyperparameter_tune_kwargs="auto",
            train_data=train_data,
            val_data=train_data,
        )

        assert mock_skip_hpo.called


def test_when_time_limit_is_capped_with_aux_params_then_time_limit_is_adjusted(temp_model_path, train_data):
    model = ConcreteTimeSeriesModel(
        path=temp_model_path,
        hyperparameters={AG_ARGS_FIT: {"max_time_limit": 5}},
    )
    with mock.patch.object(model, "_fit") as mock_internal_fit:
        model.fit(train_data=train_data, time_limit=10)
        mock_internal_fit.assert_called_once()
        assert mock_internal_fit.call_args.kwargs["time_limit"] == 5


def test_when_max_time_limit_ratio_is_provided_with_aux_params_then_time_limit_is_adjusted(
    temp_model_path, train_data
):
    model = ConcreteTimeSeriesModel(
        path=temp_model_path,
        hyperparameters={AG_ARGS_FIT: {"max_time_limit_ratio": 0.8}},
    )
    with mock.patch.object(model, "_fit") as mock_internal_fit:
        model.fit(train_data=train_data, time_limit=10)
        mock_internal_fit.assert_called_once()
        pytest.approx(mock_internal_fit.call_args.kwargs["time_limit"], 0.8 * 10)


def test_when_model_is_fit_with_time_limit_less_than_zero_then_error_is_raised(temp_model_path, train_data):
    model = ConcreteTimeSeriesModel(path=temp_model_path)
    with pytest.raises(TimeLimitExceeded):
        model.fit(train_data=train_data, time_limit=-1)


def test_when_convert_to_refit_full_via_copy_called_then_output_is_correct(temp_model_path, train_data):
    model = ConcreteTimeSeriesModel(path=temp_model_path)
    model.fit(train_data=train_data)

    copied_model = model.convert_to_refit_full_via_copy()

    assert isinstance(copied_model, ConcreteTimeSeriesModel)
    assert copied_model.path == model.path + REFIT_FULL_SUFFIX


def test_when_model_predicts_then_columns_have_correct_order(temp_model_path, train_data):
    model = ConcreteTimeSeriesModel(path=temp_model_path)
    model.fit(train_data=train_data)
    predictions = model.predict(train_data)

    assert predictions.columns.tolist() == ["mean"] + [str(q) for q in model.quantile_levels]
