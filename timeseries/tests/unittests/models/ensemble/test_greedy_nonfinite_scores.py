"""Greedy ensemble selection when every candidate regret is non-finite.

Ordinary constant / zero-demand items make MASE (in-sample scale 0) and the
default WQL (forecast-horizon sum 0) return NaN. On AutoGluon master,
``EnsembleSelection`` then dies inside ``RandomState.choice([])``. After the
core guard, ``GreedyEnsemble.fit`` / ``PerItemGreedyEnsemble.fit`` raise a
clear ``ValueError`` instead of inventing a winner.

Do not route this through ``EnsembleComposer``: its ``fit`` try/except
swallows the error and skips the ensemble, so a predictor-level crash is not
the user-facing behavior.
"""

import numpy as np
import pandas as pd
import pytest

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.ensemble import GreedyEnsemble, PerItemGreedyEnsemble

from ...common import get_data_frame_with_item_index


PREDICTION_LENGTH = 2
QUANTILE_COLUMNS = ["mean", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]


def _zero_demand_item(length: int = 12) -> TimeSeriesDataFrame:
    """Tiny intermittent / all-zero item (daily, longer than MASE seasonality)."""
    data = get_data_frame_with_item_index(["SKU"], data_length=length, freq="D", start_date="2022-01-01")
    data["target"] = 0.0
    return data


def _horizon_forecasts(data: TimeSeriesDataFrame, value: float = 0.0) -> TimeSeriesDataFrame:
    """Forecast aligned with the last ``prediction_length`` rows (the val window)."""
    horizon_index = data.slice_by_timestep(-PREDICTION_LENGTH, None).index
    return TimeSeriesDataFrame(
        pd.DataFrame(
            np.full((len(horizon_index), len(QUANTILE_COLUMNS)), value),
            index=horizon_index,
            columns=QUANTILE_COLUMNS,
        )
    )


def _zero_demand_ensemble_inputs():
    data = _zero_demand_item()
    # SeasonalNaive / ZeroModel on an all-zero item emit a zero forecast.
    zero_forecast = _horizon_forecasts(data, value=0.0)
    predictions_per_window = {
        "Naive": [zero_forecast],
        "SeasonalNaive": [zero_forecast.copy()],
    }
    return predictions_per_window, [data]


@pytest.mark.parametrize(
    "ensemble_cls, hyperparameters",
    [
        (GreedyEnsemble, {"ensemble_size": 2}),
        (PerItemGreedyEnsemble, {"ensemble_size": 2, "n_jobs": 1}),
    ],
)
@pytest.mark.parametrize("eval_metric", [None, "WQL", "MASE"])
def test_when_all_ensemble_scores_nonfinite_then_fit_raises_clear_error(ensemble_cls, hyperparameters, eval_metric):
    """Zero-demand item + MASE / default WQL: all regrets NaN, then a clear ValueError.

    Master ``EnsembleSelection`` raised ``ValueError`` from ``RandomState.choice([])``
    (``'a' cannot be empty unless no samples are taken``). After the guard this is
    ``ValueError: all ensemble scores non-finite``.
    """
    predictions_per_window, data_per_window = _zero_demand_ensemble_inputs()
    ensemble = ensemble_cls(
        prediction_length=PREDICTION_LENGTH,
        eval_metric=eval_metric,
        hyperparameters=hyperparameters,
        freq=data_per_window[0].freq,
    )
    with pytest.raises(ValueError, match="all ensemble scores non-finite"):
        ensemble.fit(
            predictions_per_window=predictions_per_window,
            data_per_window=data_per_window,
        )


def test_when_constant_series_and_mase_then_greedy_ensemble_fit_raises_clear_error():
    """MASE is undefined on a constant item (in-sample seasonal error 0)."""
    data = get_data_frame_with_item_index(["SKU"], data_length=12, freq="D", start_date="2022-01-01")
    data["target"] = 5.0
    forecast = _horizon_forecasts(data, value=5.0)
    ensemble = GreedyEnsemble(
        prediction_length=PREDICTION_LENGTH,
        eval_metric="MASE",
        hyperparameters={"ensemble_size": 2},
        freq=data.freq,
    )
    with pytest.raises(ValueError, match="all ensemble scores non-finite"):
        ensemble.fit(
            predictions_per_window={"Naive": [forecast], "SeasonalNaive": [forecast.copy()]},
            data_per_window=[data],
        )
