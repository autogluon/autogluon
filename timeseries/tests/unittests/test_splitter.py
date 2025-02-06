import pytest

from autogluon.timeseries.splitter import ExpandingWindowSplitter

from .common import DATAFRAME_WITH_COVARIATES, DATAFRAME_WITH_STATIC, DUMMY_VARIABLE_LENGTH_TS_DATAFRAME


def test_when_splitter_splits_then_underlying_data_is_not_copied():
    splitter = ExpandingWindowSplitter(prediction_length=3, num_val_windows=2)
    original_df = DATAFRAME_WITH_STATIC.copy()
    for train_fold, val_fold in splitter.split(original_df):
        assert train_fold._is_view
        assert val_fold._is_view or val_fold.values.data == original_df.values.data


def test_when_static_features_are_present_then_static_features_are_preserved():
    original_df = DATAFRAME_WITH_STATIC.copy()
    splitter = ExpandingWindowSplitter(prediction_length=3, num_val_windows=2)

    for train_fold, val_fold in splitter.split(original_df):
        assert train_fold.static_features.equals(original_df.static_features)
        assert val_fold.static_features.equals(original_df.static_features)


def test_when_covariates_are_present_then_covariates_are_preserved():
    original_df = DATAFRAME_WITH_COVARIATES.copy()
    original_columns = original_df.columns
    splitter = ExpandingWindowSplitter(prediction_length=3, num_val_windows=2)

    for train_fold, val_fold in splitter.split(original_df):
        assert train_fold.columns.equals(original_columns)
        assert val_fold.columns.equals(original_columns)


@pytest.mark.parametrize("num_val_windows", [1, 2])
@pytest.mark.parametrize("prediction_length", [1, 3])
@pytest.mark.parametrize("val_step_size", [1, 3])
def test_when_splitter_splits_then_all_folds_have_expected_length(prediction_length, num_val_windows, val_step_size):
    original_df = DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.copy()
    original_lengths = original_df.num_timesteps_per_item()
    splitter = ExpandingWindowSplitter(
        prediction_length=prediction_length, num_val_windows=num_val_windows, val_step_size=val_step_size
    )

    for window_idx, (train_fold, val_fold) in enumerate(splitter.split(original_df)):
        for item_id in original_df.item_ids:
            expected_val_length = original_lengths.loc[item_id] - (num_val_windows - 1 - window_idx) * val_step_size
            assert expected_val_length == len(val_fold.loc[item_id])

            expected_train_length = expected_val_length - prediction_length
            assert expected_train_length == len(train_fold.loc[item_id])
