import os

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
        model_path_map = {"MyModel": "models/MyModel"}
        cache.put(df, None, {"MyModel": preds}, {"MyModel": 0.5}, model_path_map=model_path_map)

        cached_preds, cached_times = cache.get(df_other, None, model_path_map=model_path_map)

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
        model_path_map = {"MyModel": "models/MyModel"}
        cache.put(df, None, {"MyModel": preds}, {"MyModel": 0.5}, model_path_map=model_path_map)

        cached_preds, cached_times = cache.get(df_other, None, model_path_map=model_path_map)

        assert not cached_preds
        assert not cached_times

    def test_when_cache_cleared_then_file_is_removed(self, tmp_path):
        df = DATAFRAME_WITH_COVARIATES
        preds = get_prediction_for_df(df)

        cache = FileBasedPredictionCache(str(tmp_path))
        model_path_map = {"MyModel": "models/MyModel"}
        cache.put(df, None, {"MyModel": preds}, {"MyModel": 0.5}, model_path_map=model_path_map)

        cache.clear()

        assert not cache.cache_dir.exists()

    def test_when_model_is_identical_then_cache_is_hit(self, tmp_path):
        df = DATAFRAME_WITH_COVARIATES
        preds = get_prediction_for_df(df)
        model_name = "MyModel"
        model_path = tmp_path / model_name
        model_path.touch()

        cache = FileBasedPredictionCache(str(tmp_path))
        model_path_map = {model_name: model_name}

        cache.put(df, None, {model_name: preds}, {model_name: 0.5}, model_path_map=model_path_map)

        # Check that predictions can be retrieved immediately
        cached_preds, _ = cache.get(df, None, model_path_map=model_path_map)
        cached_model_preds = cached_preds.get(model_name)
        assert cached_model_preds is not None
        assert cached_model_preds.equals(preds)

    def test_when_model_is_updated_then_cache_is_invalidated(self, tmp_path):
        df = DATAFRAME_WITH_COVARIATES
        preds = get_prediction_for_df(df)
        model_name = "MyModel"
        model_path = tmp_path / model_name
        model_path.touch()

        cache = FileBasedPredictionCache(str(tmp_path))
        model_path_map = {model_name: model_name}

        cache.put(df, None, {model_name: preds}, {model_name: 0.5}, model_path_map=model_path_map)

        # "Update" the model by updating its modification time
        current_stat = model_path.stat()
        os.utime(model_path, (current_stat.st_atime, current_stat.st_mtime + 1))

        cached_preds_after_update, _ = cache.get(df, None, model_path_map=model_path_map)
        assert model_name not in cached_preds_after_update
