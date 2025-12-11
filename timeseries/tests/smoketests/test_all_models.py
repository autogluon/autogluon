from typing import Any

import numpy as np
import pandas as pd
import pytest
from packaging.version import Version

from autogluon.timeseries import TimeSeriesPredictor
from autogluon.timeseries.dataset import TimeSeriesDataFrame

TARGET_COLUMN = "custom_target"
ITEM_IDS = ["Z", "A", "1", "C"]


def generate_train_and_test_data(
    prediction_length: int = 1,
    freq: str = "h",
    start_time: pd.Timestamp = "2020-01-05 15:37",
    use_known_covariates: bool = False,
    use_past_covariates: bool = False,
    use_static_features_continuous: bool = False,
    use_static_features_categorical: bool = False,
) -> tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
    min_length = prediction_length * 6
    length_per_item = {item_id: np.random.randint(min_length, min_length + 10) for item_id in ITEM_IDS}
    df_per_item = []
    for idx, (item_id, length) in enumerate(length_per_item.items()):
        start = pd.Timestamp(start_time) + (idx + 1) * pd.tseries.frequencies.to_offset(freq)
        timestamps = pd.date_range(start=start, periods=length, freq=freq)
        index = pd.MultiIndex.from_product(
            [(item_id,), timestamps], names=[TimeSeriesDataFrame.ITEMID, TimeSeriesDataFrame.TIMESTAMP]
        )
        columns = {TARGET_COLUMN: np.random.normal(size=length)}
        if use_known_covariates:
            columns["known_A"] = np.random.choice(["foo", "bar", "baz"], size=length)
            columns["known_B"] = np.random.normal(size=length)
        if use_past_covariates:
            columns["past_A"] = np.random.choice(["foo", "bar", "baz"], size=length)
            columns["past_B"] = np.random.normal(size=length)
            columns["past_C"] = np.random.normal(size=length)
        df_per_item.append(pd.DataFrame(columns, index=index))

    df = TimeSeriesDataFrame(pd.concat(df_per_item))

    if use_static_features_categorical or use_static_features_continuous:
        static_columns = {}
        if use_static_features_categorical:
            static_columns["static_A"] = np.random.choice(["foo", "bar", "baz"], size=len(ITEM_IDS))
        if use_static_features_continuous:
            static_columns["static_B"] = np.random.normal(size=len(ITEM_IDS))
        static_df = pd.DataFrame(static_columns, index=ITEM_IDS)
        df.static_features = static_df

    train_data = df.slice_by_timestep(None, -prediction_length)
    test_data = df
    return train_data, test_data


DUMMY_MODEL_HPARAMS = {
    "max_epochs": 1,
    "num_batches_per_epoch": 1,
    "use_fallback_model": False,
}

ALL_MODELS = {
    "ADIDA": DUMMY_MODEL_HPARAMS,
    "Average": DUMMY_MODEL_HPARAMS,
    "Chronos": [
        {"model_path": "autogluon/chronos-t5-tiny"},
        {"model_path": "autogluon/chronos-t5-tiny", "fine_tune": True, "fine_tune_steps": 1},
        {"model_path": "autogluon/chronos-bolt-tiny"},
        {"model_path": "autogluon/chronos-bolt-tiny", "fine_tune": True, "fine_tune_steps": 1},
    ],
    "Croston": DUMMY_MODEL_HPARAMS,
    "DLinear": DUMMY_MODEL_HPARAMS,
    "DeepAR": DUMMY_MODEL_HPARAMS,
    "DirectTabular": DUMMY_MODEL_HPARAMS,
    "DynamicOptimizedTheta": DUMMY_MODEL_HPARAMS,
    "IMAPA": DUMMY_MODEL_HPARAMS,
    "AutoETS": {**DUMMY_MODEL_HPARAMS, "model": "ANN"},
    "NPTS": DUMMY_MODEL_HPARAMS,
    "Naive": DUMMY_MODEL_HPARAMS,
    "PatchTST": DUMMY_MODEL_HPARAMS,
    "PerStepTabular": {**DUMMY_MODEL_HPARAMS, "model_name": "DUMMY"},
    "RecursiveTabular": DUMMY_MODEL_HPARAMS,
    "SeasonalAverage": DUMMY_MODEL_HPARAMS,
    "SeasonalNaive": DUMMY_MODEL_HPARAMS,
    "SimpleFeedForward": DUMMY_MODEL_HPARAMS,
    "TemporalFusionTransformer": DUMMY_MODEL_HPARAMS,
    "TiDE": DUMMY_MODEL_HPARAMS,
    "Toto": {"num_samples": 5, "device": "cpu"},
    "WaveNet": DUMMY_MODEL_HPARAMS,
    "Zero": DUMMY_MODEL_HPARAMS,
    # Override default hyperparameters for faster training
    "AutoARIMA": {"max_p": 2, "use_fallback_model": False},
}


def assert_leaderboard_contains_all_models(
    leaderboard: pd.DataFrame,
    hyperparameters: dict[str, Any],
    include_ensemble: bool = True,
):
    """Compare the leaderboard to a set of hyperparameters provided to AutoGluon-TimeSeries,
    asserting that every model that results from the hyperparameters is present in the leaderboard.
    If include_ensemble is True, also assert that the ensemble is present in the leaderboard.
    """
    # flatten the hyperparameters dict (nested list of dicts will mean multiple models)
    expected_models = []
    for k, v in hyperparameters.items():
        v = v if isinstance(v, list) else [v]
        for _ in v:
            expected_models.append(k)

    if include_ensemble:
        expected_models.append("WeightedEnsemble")

    leaderboard_models = list(leaderboard["model"])
    failed_models = []
    for model in expected_models:
        match = next((m for m in leaderboard_models if m.startswith(model)), None)
        if match is None:
            failed_models.append(match)
        else:
            leaderboard_models.remove(match)

    assert len(failed_models) == 0, f"Failed models: {failed_models}"


# TODO: Some models, such as local models, do not change behavior when past / known /
# static features are provided. We could omit them from these tests.
@pytest.mark.parametrize("use_past_covariates", [True, False])
@pytest.mark.parametrize("use_known_covariates", [True, False])
@pytest.mark.parametrize("use_static_features_categorical", [True, False])
def test_all_models_can_handle_all_covariates(
    use_known_covariates,
    use_past_covariates,
    use_static_features_categorical,
):
    prediction_length = 5
    train_data, test_data = generate_train_and_test_data(
        prediction_length=prediction_length,
        use_known_covariates=use_known_covariates,
        use_past_covariates=use_past_covariates,
        use_static_features_continuous=False,
        use_static_features_categorical=use_static_features_categorical,
    )

    known_covariates_names = [col for col in train_data if col.startswith("known_")]

    predictor = TimeSeriesPredictor(
        target=TARGET_COLUMN,
        prediction_length=prediction_length,
        known_covariates_names=known_covariates_names if len(known_covariates_names) > 0 else None,
        eval_metric="WQL",
    )
    predictor.fit(train_data, hyperparameters=ALL_MODELS)
    predictor.evaluate(test_data)
    leaderboard = predictor.leaderboard(test_data)

    assert_leaderboard_contains_all_models(leaderboard, hyperparameters=ALL_MODELS)

    known_covariates = test_data.slice_by_timestep(-prediction_length, None)[known_covariates_names]
    predictions = predictor.predict(train_data, known_covariates=known_covariates)

    future_test_data = test_data.slice_by_timestep(-prediction_length, None)

    assert predictions.index.equals(future_test_data.index)


@pytest.mark.parametrize(
    "freq",
    [
        "YE",
        "QE",
        "SME",
        "W",
        "2D",
        "B",
        "bh",
        "4h",
        "min",
        "100s",
    ],
)
@pytest.mark.parametrize(
    "hyperparameters",
    [
        {"Chronos": {"model_path": "autogluon/chronos-bolt-tiny"}},
        {"DLinear": DUMMY_MODEL_HPARAMS},
        {"DeepAR": DUMMY_MODEL_HPARAMS},
        {"DirectTabular": DUMMY_MODEL_HPARAMS},
        {"AutoETS": {**DUMMY_MODEL_HPARAMS, "model": "AAA"}},
        {"NPTS": DUMMY_MODEL_HPARAMS},
        {"Naive": DUMMY_MODEL_HPARAMS},
        {"PatchTST": DUMMY_MODEL_HPARAMS},
        {"RecursiveTabular": DUMMY_MODEL_HPARAMS},
        {"SeasonalAverage": DUMMY_MODEL_HPARAMS},
        {"SeasonalNaive": DUMMY_MODEL_HPARAMS},
        {"SimpleFeedForward": DUMMY_MODEL_HPARAMS},
        {"TemporalFusionTransformer": DUMMY_MODEL_HPARAMS},
        {"TiDE": DUMMY_MODEL_HPARAMS},
        {"WaveNet": DUMMY_MODEL_HPARAMS},
        {"Zero": DUMMY_MODEL_HPARAMS},
    ],
)
def test_all_models_handle_all_pandas_frequencies(freq, hyperparameters):
    if Version(pd.__version__) < Version("2.1") and freq in ["SME", "B", "bh"]:
        pytest.skip(f"'{freq}' frequency inference not supported by pandas < 2.1")
    if Version(pd.__version__) < Version("2.2"):
        # If necessary, convert pandas 2.2+ freq strings to an alias supported by currently installed pandas version
        freq = {"ME": "M", "QE": "Q", "YE": "Y", "SME": "SM"}.get(freq, freq)

    prediction_length = 5

    train_data, test_data = generate_train_and_test_data(
        prediction_length=prediction_length,
        freq=freq,
        use_known_covariates=True,
        use_past_covariates=True,
        start_time="1990-01-01",
    )
    known_covariates_names = [col for col in train_data.columns if col.startswith("known_")]

    predictor = TimeSeriesPredictor(
        target=TARGET_COLUMN,
        prediction_length=prediction_length,
        known_covariates_names=known_covariates_names if len(known_covariates_names) > 0 else None,
    )
    predictor.fit(train_data, hyperparameters=hyperparameters)
    predictor.evaluate(test_data)
    leaderboard = predictor.leaderboard(test_data)

    assert_leaderboard_contains_all_models(
        leaderboard,
        hyperparameters=hyperparameters,
        include_ensemble=False,
    )

    known_covariates = test_data.slice_by_timestep(-prediction_length, None)[known_covariates_names]
    predictions = predictor.predict(train_data, known_covariates=known_covariates)
    future_test_data = test_data.slice_by_timestep(-prediction_length, None)

    assert predictions.index.equals(future_test_data.index)


def test_when_tuning_data_and_time_limit_are_provided_then_all_models_are_trained():
    prediction_length = 5
    hyperparameters = {"Naive": DUMMY_MODEL_HPARAMS, "Average": DUMMY_MODEL_HPARAMS}
    train_data, test_data = generate_train_and_test_data(prediction_length=prediction_length)
    predictor = TimeSeriesPredictor(prediction_length=prediction_length, target=TARGET_COLUMN)
    predictor.fit(train_data, tuning_data=train_data, hyperparameters=hyperparameters, time_limit=120)
    leaderboard = predictor.leaderboard(test_data)
    assert_leaderboard_contains_all_models(leaderboard, hyperparameters=hyperparameters)
