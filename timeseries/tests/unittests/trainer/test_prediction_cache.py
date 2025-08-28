from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.trainer.prediction_cache import FileBasedPredictionCache, compute_dataset_hash
from autogluon.timeseries.utils.forecast import make_future_data_frame

from ..common import (
    DATAFRAME_WITH_COVARIATES,
    DATAFRAME_WITH_STATIC_AND_COVARIATES,
    get_prediction_for_df,
)


class TestDatasetHashFunction:
    def _get_known_covariates_for_df(self, df: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        known_covariates = make_future_data_frame(df, prediction_length=5)
        known_covariates["cov1"] = 42
        return TimeSeriesDataFrame(known_covariates)

    def test_when_dfs_are_identical_then_identical_hash_is_computed(self):
        df = DATAFRAME_WITH_COVARIATES
        df_other = DATAFRAME_WITH_COVARIATES.copy(deep=True)
        df_other = df_other.reindex(reversed(df_other.columns), axis=1)

        assert df is not df_other
        assert compute_dataset_hash(df) == compute_dataset_hash(df_other)

    def test_when_dfs_and_known_covariates_are_identical_then_identical_hash_is_computed(self):
        df = DATAFRAME_WITH_COVARIATES
        df_other = DATAFRAME_WITH_COVARIATES.copy(deep=True)
        df_other = df_other.reindex(reversed(df_other.columns), axis=1)

        known_covariates = self._get_known_covariates_for_df(df)
        known_covariates_other = self._get_known_covariates_for_df(df_other)

        assert df is not df_other
        assert known_covariates is not known_covariates_other
        assert compute_dataset_hash(df, known_covariates) == compute_dataset_hash(df_other, known_covariates_other)

    def test_when_different_dfs_then_different_hashes_are_computed(self):
        df = DATAFRAME_WITH_COVARIATES
        df_other = DATAFRAME_WITH_COVARIATES.copy(deep=True)
        df_other.iloc[0, 0] = df_other.iloc[0, 0] + 1  # type: ignore
        assert compute_dataset_hash(df) != compute_dataset_hash(df_other)

    def test_when_identical_dfs_and_different_known_covariates_are_identical_then_different_hashes_are_computed(self):
        df = DATAFRAME_WITH_COVARIATES
        df_other = DATAFRAME_WITH_COVARIATES.copy(deep=True)
        df_other = df_other.reindex(reversed(df_other.columns), axis=1)

        known_covariates = self._get_known_covariates_for_df(df)
        known_covariates_other = self._get_known_covariates_for_df(df_other)

        known_covariates_other.iloc[0, 0] = known_covariates_other.iloc[0, 0] * 2  # type: ignore

        assert df is not df_other
        assert compute_dataset_hash(df, known_covariates) != compute_dataset_hash(df_other, known_covariates_other)

    def test_when_different_static_features_then_different_hashes_are_computed(self):
        df = DATAFRAME_WITH_STATIC_AND_COVARIATES
        df_other = df.copy(deep=True)

        assert df_other.static_features is not None
        df_other.static_features.iloc[0, 0] = df_other.static_features.iloc[0, 0] + 1  # type: ignore

        assert compute_dataset_hash(df) != compute_dataset_hash(df_other)


class TestFileBasedPredictionCache:
    def test_when_identical_dfs_are_given_then_cache_hit_returns_true(self, tmp_path):
        df = DATAFRAME_WITH_COVARIATES
        df_other = DATAFRAME_WITH_COVARIATES.copy(deep=True)

        preds = get_prediction_for_df(df)

        cache = FileBasedPredictionCache(str(tmp_path))
        cache.put(df, None, {"MyModel": preds}, {"MyModel": 0.5})

        cached_preds, cached_times = cache.get(df_other, None)

        cached_model_preds = cached_preds.get("MyModel")
        assert cached_model_preds is not None
        assert cached_times.get("MyModel") is not None
        assert cached_model_preds.equals(preds)

    def test_when_different_dfs_are_given_then_cache_hit_returns_false(self, tmp_path):
        df = DATAFRAME_WITH_COVARIATES
        df_other = DATAFRAME_WITH_COVARIATES.copy(deep=True)
        df_other.iloc[0, 0] = df_other.iloc[0, 0] + 1  # type: ignore

        preds = get_prediction_for_df(df)

        cache = FileBasedPredictionCache(str(tmp_path))
        cache.put(df, None, {"MyModel": preds}, {"MyModel": 0.5})

        cached_preds, cached_times = cache.get(df_other, None)

        assert not cached_preds
        assert not cached_times

    def test_when_cache_cleared_then_file_is_removed(self, tmp_path):
        df = DATAFRAME_WITH_COVARIATES
        preds = get_prediction_for_df(df)

        cache = FileBasedPredictionCache(str(tmp_path))
        cache.put(df, None, {"MyModel": preds}, {"MyModel": 0.5})

        cache.clear()

        expected_path = tmp_path / cache._cached_predictions_filename
        assert expected_path == cache.path
        assert not expected_path.exists()
