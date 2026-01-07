import os
import shutil
import tempfile

import pytest

from autogluon.timeseries.models import ETSModel
from autogluon.timeseries.models.multi_window import MultiWindowBacktestingModel
from autogluon.timeseries.splitter import ExpandingWindowSplitter
from autogluon.timeseries.utils.features import CovariateMetadata

from ..common import DUMMY_TS_DATAFRAME, dict_equal_primitive


def test_when_model_base_kwargs_passed_to_mw_model_then_kwargs_passed_to_base_model(temp_model_path):
    model_base_kwargs = {"hyperparameters": {"seasonality": "mul", "trend": None}, "prediction_length": 4}
    mw_model = MultiWindowBacktestingModel(
        model_base=ETSModel, model_base_kwargs=model_base_kwargs, path=temp_model_path
    )
    received_kwargs = mw_model.model_base.get_params()

    assert model_base_kwargs["prediction_length"] == received_kwargs["prediction_length"]
    assert dict_equal_primitive(received_kwargs["hyperparameters"], model_base_kwargs["hyperparameters"])


@pytest.mark.parametrize("prediction_length", [1, 3])
@pytest.mark.parametrize("num_val_windows", [1, 2])
def test_when_mw_model_trained_then_oof_predictions_and_stats_are_saved(
    multi_window_deepar_model_class,
    temp_model_path,
    prediction_length,
    num_val_windows,
):
    val_splitter = ExpandingWindowSplitter(prediction_length=prediction_length, num_val_windows=num_val_windows)
    mw_model = multi_window_deepar_model_class(
        path=temp_model_path, prediction_length=prediction_length, freq=DUMMY_TS_DATAFRAME.freq
    )
    mw_model.fit(train_data=DUMMY_TS_DATAFRAME, val_splitter=val_splitter)

    assert len(mw_model.get_oof_predictions()) == num_val_windows
    for oof_pred in mw_model.get_oof_predictions():
        assert len(oof_pred) == prediction_length * DUMMY_TS_DATAFRAME.num_items
    assert mw_model.val_score is not None
    assert mw_model.predict_time is not None


def test_when_val_data_passed_to_mw_model_fit_then_exception_is_raised(
    multi_window_deepar_model_class, temp_model_path
):
    mw_model = multi_window_deepar_model_class(path=temp_model_path)
    with pytest.raises(ValueError, match="val_data should not be passed"):
        mw_model.fit(train_data=DUMMY_TS_DATAFRAME, val_data=DUMMY_TS_DATAFRAME)


def test_when_saved_model_moved_then_model_can_be_loaded_with_updated_path(multi_window_deepar_model_class):
    original_path = tempfile.mkdtemp() + os.sep
    model = multi_window_deepar_model_class(path=original_path, freq=DUMMY_TS_DATAFRAME.freq)
    model.fit(train_data=DUMMY_TS_DATAFRAME)
    model.save()
    new_path = tempfile.mkdtemp() + os.sep
    shutil.move(os.path.join(original_path, model.name), new_path)
    loaded_model = model.load(os.path.join(new_path, model.name) + os.sep)
    assert loaded_model.path.startswith(new_path)
    assert loaded_model.most_recent_model.path.startswith(new_path)

    shutil.rmtree(original_path)
    shutil.rmtree(new_path)


def test_when_multi_window_model_created_then_regressor_and_scaler_are_created_only_for_base_model(
    multi_window_deepar_model_class,
):
    data = DUMMY_TS_DATAFRAME.copy()
    data["feat1"] = range(len(data))
    model = multi_window_deepar_model_class(
        freq=data.freq,
        hyperparameters={"target_scaler": "standard", "covariate_regressor": "LR"},
        covariate_metadata=CovariateMetadata(known_covariates_real=["feat1"]),
    )
    model.fit(train_data=data, time_limit=5.0)
    assert model.covariate_regressor is None
    assert model.target_scaler is None
    assert model.most_recent_model.covariate_regressor is not None
    assert model.most_recent_model.target_scaler is not None


def test_when_score_and_cache_oof_called_then_val_data_predictions_appended(
    multi_window_deepar_model_class, temp_model_path
):
    num_val_windows = 2
    val_splitter = ExpandingWindowSplitter(prediction_length=1, num_val_windows=num_val_windows)
    mw_model = multi_window_deepar_model_class(path=temp_model_path, freq=DUMMY_TS_DATAFRAME.freq)

    # Fit on train_data with multi-window
    mw_model.fit(train_data=DUMMY_TS_DATAFRAME, val_splitter=val_splitter)

    # Should have 2 OOF predictions from train_data windows
    assert len(mw_model.get_oof_predictions()) == num_val_windows

    # Call score_and_cache_oof with val_data
    val_data = DUMMY_TS_DATAFRAME.slice_by_timestep(-5, None)
    mw_model.score_and_cache_oof(val_data, store_val_score=True)

    # Should now have 3 OOF predictions (2 from train + 1 from val_data)
    assert len(mw_model.get_oof_predictions()) == num_val_windows + 1
    assert mw_model.val_score is not None
