"""Unit tests for TimeSeriesPredictor.save_space"""

import os

import pytest

from autogluon.timeseries.predictor import TimeSeriesPredictor

from .common import DUMMY_TS_DATAFRAME

DUMMY_HYPERPARAMETERS = {"SeasonalNaive": {"n_jobs": 1}, "Average": {"n_jobs": 1}}

CACHED_DATA_FILES = ["train.pkl", "val.pkl", "val_data_per_window.pkl"]


def _fit_predictor(path):
    predictor = TimeSeriesPredictor(path=path, prediction_length=2)
    predictor.fit(train_data=DUMMY_TS_DATAFRAME, hyperparameters=DUMMY_HYPERPARAMETERS)
    return predictor


def test_when_save_space_called_then_cached_data_files_are_deleted(temp_model_path):
    predictor = _fit_predictor(temp_model_path)
    path_data = predictor._trainer.path_data

    # at least one cached data file should exist after fit
    assert any(os.path.exists(os.path.join(path_data, f)) for f in CACHED_DATA_FILES)

    predictor.save_space()

    for file_name in CACHED_DATA_FILES:
        assert not os.path.exists(os.path.join(path_data, file_name))


def test_when_save_space_called_then_prediction_still_works(temp_model_path):
    predictor = _fit_predictor(temp_model_path)
    predictions_before = predictor.predict(DUMMY_TS_DATAFRAME)

    predictor.save_space()
    predictions_after = predictor.predict(DUMMY_TS_DATAFRAME)

    # inference accuracy is unaffected
    assert predictions_before.equals(predictions_after)


def test_when_save_space_called_on_unfit_predictor_then_raises(temp_model_path):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    with pytest.raises(AssertionError):
        predictor.save_space()
