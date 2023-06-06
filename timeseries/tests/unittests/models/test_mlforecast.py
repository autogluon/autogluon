import numpy as np
import pandas as pd
import pytest

from memory_profiler import memory_usage

from autogluon.timeseries.models.autogluon_tabular.mlforecast import RecursiveTabularModel, MLFMemoryUsage
from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator

from ..common import get_data_frame_with_variable_lengths

TESTABLE_MODELS = [
    RecursiveTabularModel,
]


@pytest.mark.parametrize("known_covariates_names", [["known_1", "known_2"], []])
@pytest.mark.parametrize("static_features_names", [["cat_1"], []])
@pytest.mark.parametrize("differences", [[2, 3], []])
@pytest.mark.parametrize("lags", [[1, 2, 5], [4]])
def test_when_covariates_and_features_present_then_feature_df_shape_is_correct(
    temp_model_path, known_covariates_names, static_features_names, differences, lags
):
    item_id_to_length = {1: 10, 5: 20, 2: 30}
    data = get_data_frame_with_variable_lengths(item_id_to_length, covariates_names=known_covariates_names)
    if static_features_names:
        columns = {k: np.random.normal(size=len(item_id_to_length)) for k in static_features_names}
        data.static_features = pd.DataFrame(columns, index=data.item_ids)

    feat_gen = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    data = feat_gen.fit_transform(data)
    # Initialize model._target_lag_indices and model._time_features from freq
    model = RecursiveTabularModel(
        freq=data.freq,
        path=temp_model_path,
        metadata=feat_gen.covariate_metadata,
        hyperparameters={"differences": differences, "lags": lags},
    )
    model.fit(train_data=data, time_limit=2)
    X, y = model._get_features_dataframe(data)
    expected_num_features = (
        len(lags) + len(known_covariates_names) + len(static_features_names) + len(model._get_date_features(data.freq))
    )
    expected_num_rows = len(data) - sum(differences) * data.num_items  # sum(differences) rows  dropped per item
    assert X.shape == (expected_num_rows, expected_num_features)
    assert y.shape == (expected_num_rows,)


@pytest.mark.parametrize("ts_length", [1000, 10_000])
@pytest.mark.parametrize("num_features", [2, 20])
def test_when_memory_usage_is_estimated_then_actual_mem_usage_is_approximately_equal(
    ts_length,
    num_features,
    temp_model_path,
):
    data = get_data_frame_with_variable_lengths({i: ts_length for i in range(50)})
    model = RecursiveTabularModel(
        freq=data.freq,
        path=temp_model_path,
        hyperparameters={"lags": list(range(1, num_features + 1)), "date_features": []},
    )
    predicted_mem_usage = MLFMemoryUsage()._estimate_memory_usage(num_rows=len(data), num_features=num_features)

    def fit_model():
        model.fit(train_data=data)

    actual_mem_usage = max(memory_usage(fit_model))
    assert np.isclose(predicted_mem_usage, actual_mem_usage, rtol=0.25)
