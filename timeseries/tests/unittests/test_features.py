import numpy as np
import pandas as pd
import pytest

from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator

from .common import get_data_frame_with_variable_lengths


@pytest.mark.parametrize("known_covariates_names", [["known_1", "known_2"], []])
@pytest.mark.parametrize("past_covariates_names", [["past_1", "past_2", "past_3"], []])
@pytest.mark.parametrize("static_features_cat", [["cat_1"], []])
@pytest.mark.parametrize("static_features_real", [["real_1", "real_3"], []])
def test_when_covariates_present_in_data_then_they_are_included_in_metadata(
    known_covariates_names, past_covariates_names, static_features_cat, static_features_real
):
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    item_id_to_length = {1: 10, 5: 20, 2: 30}
    data = get_data_frame_with_variable_lengths(
        item_id_to_length, covariates_names=known_covariates_names + past_covariates_names
    )
    if static_features_cat or static_features_real:
        cat_dict = {k: np.random.choice(["foo", "bar"], size=len(item_id_to_length)) for k in static_features_cat}
        real_dict = {k: np.random.normal(size=len(item_id_to_length)) for k in static_features_real}
        data.static_features = pd.DataFrame({**cat_dict, **real_dict}, index=data.item_ids)
    feat_generator.fit(data)
    metadata = feat_generator.covariate_metadata

    assert metadata.known_covariates_real == known_covariates_names
    assert metadata.past_covariates_real == past_covariates_names
    assert metadata.static_features_cat == static_features_cat
    assert metadata.static_features_real == static_features_real
