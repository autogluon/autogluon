from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from autogluon.timeseries import TimeSeriesPredictor
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.utils.datetime.seasonality import DEFAULT_SEASONALITIES

TARGET_COLUMN = "custom_target"
ITEM_IDS = ["Z", "A", "1", "C"]


def generate_train_and_test_data(
    prediction_length: int = 1,
    freq: str = "H",
    use_known_covariates: bool = False,
    use_past_covariates: bool = False,
    use_static_features_continuous: bool = False,
    use_static_features_categorical: bool = False,
) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
    min_length = prediction_length * 6
    length_per_item = {item_id: np.random.randint(min_length, min_length + 10) for item_id in ITEM_IDS}
    df_per_item = []
    for idx, (item_id, length) in enumerate(length_per_item.items()):
        start = pd.Timestamp("2020-01-05 15:37") + (idx + 1) * pd.tseries.frequencies.to_offset(freq)
        timestamps = pd.date_range(start=start, periods=length, freq=freq)
        index = pd.MultiIndex.from_product([(item_id,), timestamps], names=[ITEMID, TIMESTAMP])
        columns = {TARGET_COLUMN: np.random.normal(size=length)}
        if use_known_covariates:
            columns["known_A"] = np.random.randint(0, 10, size=length)
            columns["known_B"] = np.random.normal(size=length)
        if use_past_covariates:
            columns["past_A"] = np.random.randint(0, 10, size=length)
            columns["past_B"] = np.random.normal(size=length)
            columns["past_C"] = np.random.normal(size=length)
        df_per_item.append(pd.DataFrame(columns, index=index))

    df = TimeSeriesDataFrame(pd.concat(df_per_item))

    if use_static_features_categorical or use_static_features_continuous:
        static_columns = {}
        if use_static_features_categorical:
            static_columns["static_A"] = np.random.choice(["foo", "bar", "bazz"], size=len(ITEM_IDS))
        if use_static_features_continuous:
            static_columns["static_B"] = np.random.normal(size=len(ITEM_IDS))
        static_df = pd.DataFrame(static_columns, index=ITEM_IDS)
        df.static_features = static_df

    train_data = df.slice_by_timestep(None, -prediction_length)
    test_data = df
    return train_data, test_data


DUMMY_MODEL_HPARAMS = {"epochs": 1, "num_batches_per_epoch": 1, "use_fallback_model": False}

ALL_MODELS = {
    "Average": DUMMY_MODEL_HPARAMS,
    "DLinear": DUMMY_MODEL_HPARAMS,
    "DeepAR": DUMMY_MODEL_HPARAMS,
    "DirectTabular": DUMMY_MODEL_HPARAMS,
    "DynamicOptimizedTheta": DUMMY_MODEL_HPARAMS,
    "ETS": DUMMY_MODEL_HPARAMS,
    "NPTS": DUMMY_MODEL_HPARAMS,
    "Naive": DUMMY_MODEL_HPARAMS,
    "PatchTST": DUMMY_MODEL_HPARAMS,
    "RecursiveTabular": DUMMY_MODEL_HPARAMS,
    "SeasonalAverage": DUMMY_MODEL_HPARAMS,
    "SeasonalNaive": DUMMY_MODEL_HPARAMS,
    "SimpleFeedForward": DUMMY_MODEL_HPARAMS,
    "TemporalFusionTransformer": DUMMY_MODEL_HPARAMS,
    "Theta": DUMMY_MODEL_HPARAMS,
    # Override default hyperparameters for faster training
    "AutoARIMA": {"max_p": 2, "use_fallback_model": False},
}


def assert_leaderboard_contains_all_models(leaderboard: pd.DataFrame, include_ensemble: bool = True):
    expected_models = set(ALL_MODELS)
    if include_ensemble:
        expected_models = expected_models.union({"WeightedEnsemble"})
    failed_models = expected_models.difference(set(leaderboard["model"]))
    assert len(failed_models) == 0, f"Failed models: {failed_models}"


@pytest.mark.parametrize("use_past_covariates", [True, False])
@pytest.mark.parametrize("use_known_covariates", [True, False])
@pytest.mark.parametrize("use_static_features_continuous", [True, False])
@pytest.mark.parametrize("use_static_features_categorical", [True, False])
@pytest.mark.parametrize("eval_metric", ["WQL", "MASE"])
def test_all_models_can_handle_all_covariates(
    use_known_covariates,
    use_past_covariates,
    use_static_features_continuous,
    use_static_features_categorical,
    eval_metric,
):
    prediction_length = 5
    train_data, test_data = generate_train_and_test_data(
        prediction_length=prediction_length,
        use_known_covariates=use_known_covariates,
        use_past_covariates=use_past_covariates,
        use_static_features_continuous=use_static_features_continuous,
        use_static_features_categorical=use_static_features_categorical,
    )

    known_covariates_names = [col for col in train_data if col.startswith("known_")]

    predictor = TimeSeriesPredictor(
        target=TARGET_COLUMN,
        prediction_length=prediction_length,
        known_covariates_names=known_covariates_names if len(known_covariates_names) > 0 else None,
        eval_metric=eval_metric,
    )
    predictor.fit(train_data, hyperparameters=ALL_MODELS)
    predictor.score(test_data)
    leaderboard = predictor.leaderboard(test_data)

    assert_leaderboard_contains_all_models(leaderboard)

    known_covariates = test_data.slice_by_timestep(-prediction_length, None)[known_covariates_names]
    predictions = predictor.predict(train_data, known_covariates=known_covariates)

    future_test_data = test_data.slice_by_timestep(-prediction_length, None)

    assert predictions.index.equals(future_test_data.index)


@pytest.mark.parametrize("freq", DEFAULT_SEASONALITIES.keys())
def test_all_models_handle_all_pandas_frequencies(freq):
    freq_str = f"2{freq}"
    prediction_length = 5

    train_data, test_data = generate_train_and_test_data(
        prediction_length=prediction_length,
        freq=freq_str,
        use_known_covariates=True,
        use_past_covariates=True,
    )
    known_covariates_names = [col for col in train_data if col.startswith("known_")]

    predictor = TimeSeriesPredictor(
        target=TARGET_COLUMN,
        prediction_length=prediction_length,
        known_covariates_names=known_covariates_names if len(known_covariates_names) > 0 else None,
    )
    predictor.fit(train_data, hyperparameters=ALL_MODELS)
    predictor.score(test_data)
    leaderboard = predictor.leaderboard(test_data)

    assert_leaderboard_contains_all_models(leaderboard)

    known_covariates = test_data.slice_by_timestep(-prediction_length, None)[known_covariates_names]
    predictions = predictor.predict(train_data, known_covariates=known_covariates)

    future_test_data = test_data.slice_by_timestep(-prediction_length, None)

    assert predictions.index.equals(future_test_data.index)
