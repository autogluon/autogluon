import numpy as np
import pandas as pd
import pytest

from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator

from .common import DATAFRAME_WITH_STATIC, get_data_frame_with_variable_lengths


@pytest.mark.parametrize("known_covariates_cat", [["known_cat_1"], []])
@pytest.mark.parametrize("known_covariates_real", [["known_real_1", "known_real_2"], []])
@pytest.mark.parametrize("past_covariates_cat", [["past_cat_1", "past_cat_2"], []])
@pytest.mark.parametrize("past_covariates_real", [["past_real_1"], []])
@pytest.mark.parametrize("static_features_cat", [["static_cat_1"], []])
@pytest.mark.parametrize("static_features_real", [["static_real_1", "static_real_2"], []])
def test_when_covariates_present_in_data_then_they_are_included_in_metadata(
    known_covariates_cat,
    known_covariates_real,
    past_covariates_cat,
    past_covariates_real,
    static_features_cat,
    static_features_real,
):
    known_covariates_names = known_covariates_cat + known_covariates_real
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    item_id_to_length = {1: 10, 5: 20, 2: 30}
    data = get_data_frame_with_variable_lengths(item_id_to_length)
    for col in known_covariates_cat + past_covariates_cat:
        data[col] = np.random.choice(["foo", "bar", "baz"], size=len(data))
    for col in known_covariates_real + past_covariates_real:
        data[col] = np.random.normal(size=len(data))
    if static_features_cat or static_features_real:
        cat_dict = {k: np.random.choice(["foo", "bar"], size=len(item_id_to_length)) for k in static_features_cat}
        real_dict = {k: np.random.normal(size=len(item_id_to_length)) for k in static_features_real}
        data.static_features = pd.DataFrame({**cat_dict, **real_dict}, index=data.item_ids)
    feat_generator.fit(data)
    metadata = feat_generator.covariate_metadata

    assert metadata.known_covariates_cat == known_covariates_cat
    assert metadata.known_covariates_real == known_covariates_real
    assert metadata.past_covariates_cat == past_covariates_cat
    assert metadata.past_covariates_real == past_covariates_real
    assert metadata.static_features_cat == static_features_cat
    assert metadata.static_features_real == static_features_real


def test_given_duplicate_static_features_then_generator_can_fit_transform():
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=[])
    data = DATAFRAME_WITH_STATIC.copy()
    data.static_features["feat5"] = data.static_features["feat1"]
    feat_generator.fit(data)
    out = feat_generator.transform(data)
    assert isinstance(out.static_features, pd.DataFrame)
