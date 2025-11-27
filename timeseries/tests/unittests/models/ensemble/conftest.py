import itertools

import pytest

from autogluon.timeseries.models import SeasonalNaiveModel

from ...common import get_data_frame_with_item_index


@pytest.fixture(params=itertools.product([1, 2], [2, 3], [1, 3]))
def predictions_data_and_prediction_length(request, temp_model_path):
    """Shared fixture for generating OOF predictions, validation data, and model scores."""
    num_windows, num_models, prediction_length = request.param
    full_data = get_data_frame_with_item_index(["A", "B", "C"], start_date="2022-01-01", freq="D", data_length=120)
    data_per_window = []
    for window_idx in range(1, num_windows + 1):
        val_end = -(num_windows - window_idx) * prediction_length
        val_end = None if val_end == 0 else val_end
        data_per_window.append(full_data.slice_by_timestep(None, val_end))

    preds_per_window = {f"SNaive{s}": [] for s in range(1, num_models + 1)}
    for data in data_per_window:
        train_data, _ = data.train_test_split(prediction_length)
        for s in range(1, num_models + 1):
            preds_per_window[f"SNaive{s}"].append(
                SeasonalNaiveModel(
                    prediction_length=prediction_length,
                    hyperparameters={"seasonal_period": s, "n_jobs": 1},
                    path=temp_model_path,
                )
                .fit(train_data)
                .predict(train_data)
            )

    model_scores = {f"SNaive{s}": s * -0.1 for s in range(1, num_models + 1)}
    yield preds_per_window, data_per_window, model_scores, prediction_length
