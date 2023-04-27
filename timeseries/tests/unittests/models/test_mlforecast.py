from unittest import mock

import numpy as np
import pandas as pd
import pytest

from autogluon.timeseries.models.autogluon_tabular.mlforecast import RecurrentTabularModel
from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator

from ..common import get_data_frame_with_variable_lengths

TESTABLE_MODELS = [
    RecurrentTabularModel,
]


@pytest.mark.parametrize("known_covariates_names", [["known_1", "known_2"], []])
# @pytest.mark.parametrize("past_covariates_names", [["past_1", "past_2", "past_3"], []])
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
    model = RecurrentTabularModel(
        freq=data.freq,
        path=temp_model_path,
        metadata=feat_gen.covariate_metadata,
        hyperparameters={"differences": differences, "lags": lags},
    )
    model.fit(train_data=data, time_limit=2)
    df = model._get_features_dataframe(data)
    expected_num_features = (
        1  # target
        + len(lags)
        + len(known_covariates_names)
        + len(static_features_names)
        + len(model._get_date_features(data.freq))
    )
    expected_num_rows = len(data) - sum(differences) * data.num_items  # sum(differences) rows are dropped for each time series
    assert df.shape == (expected_num_rows, expected_num_features)
