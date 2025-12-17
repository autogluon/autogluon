import os
import shutil
from unittest import mock

import pytest

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries.models.chronos import Chronos2Model

from ...common import DATAFRAME_WITH_COVARIATES, DUMMY_TS_DATAFRAME, get_data_frame_with_item_index
from ..common import CHRONOS2_MODEL_PATH


class TestChronos2Inference:
    @pytest.fixture()
    def chronos2_model(self, tmp_path_factory):
        return Chronos2Model(
            prediction_length=5,
            path=str(tmp_path_factory.mktemp("chronos2")),
            hyperparameters={"model_path": CHRONOS2_MODEL_PATH},
        )

    @pytest.fixture()
    def mocked_chronos2_model(self, tmp_path_factory):
        with mock.patch("chronos.chronos2.pipeline.Chronos2Pipeline.from_pretrained") as mock_pretrained:
            mock_pipeline = mock.Mock()
            mock_pipeline.fit.return_value = mock_pipeline
            mock_pretrained.return_value = mock_pipeline

            yield Chronos2Model(
                prediction_length=5,
                path=str(tmp_path_factory.mktemp("mocked_chronos2")),
                hyperparameters={"model_path": CHRONOS2_MODEL_PATH},
            )

    def test_when_past_only_covariates_provided_then_chronos2_uses_them(self, chronos2_model):
        data = DATAFRAME_WITH_COVARIATES
        past_data = data.slice_by_timestep(None, -5)

        expected_past_covariates = set(past_data.columns.drop("target").tolist())

        chronos2_model.fit(train_data=past_data)
        with mock.patch("chronos.chronos2.pipeline.Chronos2Pipeline.predict_quantiles") as mocked_predict_quantiles:
            try:
                chronos2_model.predict(past_data)
            except ValueError as e:
                # ValueError due to return value of predict_quantiles
                assert "not enough values to unpack" in str(e)
            finally:
                mocked_predict_quantiles.assert_called_once()
                inputs = mocked_predict_quantiles.call_args.kwargs["inputs"]

                for input_dict in inputs:
                    past_covariates = set(input_dict["past_covariates"].keys())
                    assert past_covariates == expected_past_covariates
                    assert "future_covariates" not in input_dict

    def test_when_known_covariates_provided_then_chronos2_uses_them(self, chronos2_model):
        data = DATAFRAME_WITH_COVARIATES
        past_data = data.slice_by_timestep(None, -5)
        future_data = data.slice_by_timestep(-5, None)
        known_covariates = future_data.drop(columns=["target"])

        expected_past_covariates = set(past_data.columns.drop("target").tolist())
        expected_future_covariates = set(known_covariates.columns.tolist())

        chronos2_model.fit(train_data=past_data, known_covariates=known_covariates)
        with mock.patch("chronos.chronos2.pipeline.Chronos2Pipeline.predict_quantiles") as mocked_predict_quantiles:
            try:
                chronos2_model.predict(past_data, known_covariates=known_covariates)
            except ValueError as e:
                # ValueError due to return value of predict_quantiles
                assert "not enough values to unpack" in str(e)
            finally:
                mocked_predict_quantiles.assert_called_once()
                inputs = mocked_predict_quantiles.call_args.kwargs["inputs"]

                for input_dict in inputs:
                    past_covariates = set(input_dict["past_covariates"].keys())
                    future_covariates = set(input_dict["future_covariates"].keys())
                    assert past_covariates == expected_past_covariates
                    assert future_covariates == expected_future_covariates

    def test_when_model_persisted_then_pipeline_loaded(self, mocked_chronos2_model):
        mocked_chronos2_model.persist()
        assert mocked_chronos2_model._model_pipeline is not None

    def test_when_chronos2_saved_to_custom_path_then_model_can_be_loaded(self, chronos2_model, tmp_path):
        chronos2_model.fit(train_data=DUMMY_TS_DATAFRAME)
        chronos2_model.save(path=str(tmp_path))

        loaded_model = Chronos2Model.load(path=str(tmp_path))
        predictions = loaded_model.predict(DUMMY_TS_DATAFRAME)

        assert len(predictions) == DUMMY_TS_DATAFRAME.num_items * chronos2_model.prediction_length

    def test_when_predict_called_then_output_format_correct(self, chronos2_model):
        chronos2_model.fit(DUMMY_TS_DATAFRAME)
        predictions = chronos2_model.predict(DUMMY_TS_DATAFRAME)

        assert "target_name" not in predictions.columns
        assert "predictions" not in predictions.columns
        assert "mean" in predictions.columns
        for q in chronos2_model.quantile_levels:
            assert str(q) in predictions.columns

    def test_when_fine_tune_disabled_then_model_does_not_call_fit(self, mocked_chronos2_model):
        mocked_chronos2_model.fit(DUMMY_TS_DATAFRAME)

        mocked_chronos2_model._model_pipeline.fit.assert_not_called()

    def test_when_revision_provided_then_from_pretrained_is_called_with_revision(self):
        model_revision = "my-test-branch"
        model = Chronos2Model(
            hyperparameters={
                "model_path": CHRONOS2_MODEL_PATH,
                "revision": model_revision,
            },
        )
        with mock.patch("chronos.chronos2.pipeline.Chronos2Pipeline.from_pretrained") as mock_from_pretrained:
            mock_from_pretrained.return_value = mock.MagicMock()
            model.load_model_pipeline()

        mock_from_pretrained.assert_called_once()
        assert mock_from_pretrained.call_args.kwargs.get("revision") == model_revision

    def test_when_chronos_scores_oof_and_time_limit_is_exceeded_then_exception_is_raised(self, chronos2_model):
        data = get_data_frame_with_item_index(item_list=list(range(1000)), data_length=50)
        chronos2_model.fit(data)
        with pytest.raises(TimeLimitExceeded):
            chronos2_model.score_and_cache_oof(data, time_limit=0.1)


class TestChronos2FineTuning:
    @pytest.fixture()
    def mocked_fine_tunable_chronos2_model(self, tmp_path_factory):
        with mock.patch("chronos.chronos2.pipeline.Chronos2Pipeline.from_pretrained") as mock_pretrained:
            mock_pipeline = mock.Mock()
            mock_pipeline.fit.return_value = mock_pipeline
            mock_pretrained.return_value = mock_pipeline

            model = Chronos2Model(
                path=str(tmp_path_factory.mktemp("mocked_chronos2")),
                prediction_length=5,
                hyperparameters={
                    "model_path": CHRONOS2_MODEL_PATH,
                    "fine_tune": True,
                    "fine_tune_steps": 42,
                    "fine_tune_batch_size": 42,
                },
            )

            yield model

    @pytest.fixture(scope="class")
    def fine_tuned_chronos2_model(self, tmp_path_factory):
        model = Chronos2Model(
            path=str(tmp_path_factory.mktemp("fine_tuned_chronos2")),
            prediction_length=5,
            hyperparameters={
                "model_path": CHRONOS2_MODEL_PATH,
                "fine_tune": True,
                "fine_tune_steps": 2,
                "fine_tune_batch_size": 5,
            },
        )
        model.fit(DUMMY_TS_DATAFRAME)
        model.save()
        yield model

    def test_when_fine_tune_enabled_then_model_calls_fit_on_pipeline(self, mocked_fine_tunable_chronos2_model):
        mocked_fine_tunable_chronos2_model.fit(DUMMY_TS_DATAFRAME)

        mock_pipeline = mocked_fine_tunable_chronos2_model._model_pipeline

        mock_pipeline.fit.assert_called_once()
        assert mock_pipeline.fit.call_args.kwargs["num_steps"] == 42
        assert mock_pipeline.fit.call_args.kwargs["batch_size"] == 42

    def test_when_validation_data_provided_and_eval_turned_on_then_validation_inputs_passed(self, tmp_path):
        tmp_dir = tmp_path / "mocked_chronos2"
        tmp_dir.mkdir()
        model = Chronos2Model(
            path=str(tmp_dir),
            prediction_length=5,
            hyperparameters={
                "model_path": CHRONOS2_MODEL_PATH,
                "fine_tune": True,
                "fine_tune_steps": 2,
                "fine_tune_batch_size": 3,
                "eval_during_fine_tune": True,
            },
        )

        with mock.patch("chronos.chronos2.pipeline.Chronos2Pipeline.fit") as mocked_pipeline_fit:
            model.fit(train_data=DUMMY_TS_DATAFRAME, val_data=DUMMY_TS_DATAFRAME)
            mocked_pipeline_fit.assert_called_once()
            assert mocked_pipeline_fit.call_args.kwargs["validation_inputs"] is not None

    def test_when_fine_tuned_then_is_fine_tuned_flag_set(self, mocked_fine_tunable_chronos2_model):
        assert not mocked_fine_tunable_chronos2_model._is_fine_tuned

        mocked_fine_tunable_chronos2_model.fit(DUMMY_TS_DATAFRAME)

        assert mocked_fine_tunable_chronos2_model._is_fine_tuned

    def test_when_fine_tuned_then_output_dir_passed_to_fit(self, mocked_fine_tunable_chronos2_model):
        mocked_fine_tunable_chronos2_model.fit(DUMMY_TS_DATAFRAME)
        mock_pipeline = mocked_fine_tunable_chronos2_model._model_pipeline

        assert mock_pipeline.fit.call_args.kwargs["output_dir"] == mocked_fine_tunable_chronos2_model.path

    def test_when_fine_tuned_then_model_path_returns_local_path(self, fine_tuned_chronos2_model):
        assert fine_tuned_chronos2_model.model_path.startswith(fine_tuned_chronos2_model.path)
        assert "fine-tuned-ckpt" in fine_tuned_chronos2_model.model_path

    def test_when_fine_tuned_then_local_path_has_checkpoint(self, fine_tuned_chronos2_model):
        ckpt_path = fine_tuned_chronos2_model.model_path
        assert os.path.isdir(ckpt_path)
        assert os.path.exists(os.path.join(ckpt_path, "adapter_model.safetensors")) or os.path.exists(
            os.path.join(ckpt_path, "model.safetensors")
        )

    def test_when_fine_tuned_and_saved_then_model_can_be_loaded(self, fine_tuned_chronos2_model):
        loaded_model = Chronos2Model.load(path=fine_tuned_chronos2_model.path)

        assert loaded_model._is_fine_tuned
        assert loaded_model.model_path.startswith(loaded_model.path)

    def test_when_fine_tuned_model_loaded_then_can_predict(self, fine_tuned_chronos2_model):
        loaded_model = Chronos2Model.load(path=fine_tuned_chronos2_model.path)
        predictions = loaded_model.predict(DUMMY_TS_DATAFRAME)

        assert len(predictions) == DUMMY_TS_DATAFRAME.num_items * loaded_model.prediction_length
        assert "mean" in predictions.columns
        assert not predictions.isna().any().any()

    def test_when_fine_tuned_and_moved_then_model_path_updates(self, fine_tuned_chronos2_model, tmp_path):
        model = fine_tuned_chronos2_model
        original_path = model.path
        new_path = str(tmp_path / "moved")

        shutil.copytree(original_path, new_path)

        loaded_model = Chronos2Model.load(path=new_path)
        predictions = loaded_model.predict(DUMMY_TS_DATAFRAME)

        assert loaded_model._is_fine_tuned
        assert loaded_model.model_path.startswith(loaded_model.path)

        assert len(predictions) == DUMMY_TS_DATAFRAME.num_items * loaded_model.prediction_length
        assert "mean" in predictions.columns

    def test_when_covariates_provided_then_chronos2_is_fine_tuned_with_them(self, tmp_path, df_with_covariates):
        data, covariate_metadata = df_with_covariates
        past_data = data.slice_by_timestep(None, -5)

        expected_past_covariates = set(covariate_metadata.covariates)
        expected_future_covariates = set(covariate_metadata.known_covariates)

        tmp_dir = tmp_path / "mocked_chronos2"
        tmp_dir.mkdir()
        model = Chronos2Model(
            path=str(tmp_dir),
            prediction_length=5,
            hyperparameters={
                "model_path": CHRONOS2_MODEL_PATH,
                "fine_tune": True,
                "fine_tune_steps": 2,
                "fine_tune_batch_size": 3,
            },
            covariate_metadata=covariate_metadata,
        )

        with mock.patch("chronos.chronos2.pipeline.Chronos2Pipeline.fit") as mocked_pipeline_fit:
            model.fit(past_data)
            mocked_pipeline_fit.assert_called_once()
            inputs = mocked_pipeline_fit.call_args.kwargs["inputs"]

            for input_dict in inputs:
                past_covariates = set(input_dict["past_covariates"].keys())
                future_covariates = set(input_dict["future_covariates"].keys())

                assert past_covariates == expected_past_covariates
                assert future_covariates == expected_future_covariates

    @pytest.mark.parametrize(
        "disable_past,disable_known",
        [(True, False), (False, True), (True, True)],
    )
    def test_when_covariates_disabled_then_not_used_during_fit(
        self, tmp_path, df_with_covariates, disable_past, disable_known
    ):
        data, covariate_metadata = df_with_covariates

        tmp_dir = tmp_path / "mocked_chronos2"
        tmp_dir.mkdir()
        model = Chronos2Model(
            path=str(tmp_dir),
            prediction_length=5,
            hyperparameters={
                "model_path": CHRONOS2_MODEL_PATH,
                "fine_tune": True,
                "fine_tune_steps": 2,
                "fine_tune_batch_size": 3,
                "disable_past_covariates": disable_past,
                "disable_known_covariates": disable_known,
            },
            covariate_metadata=covariate_metadata,
        )

        with mock.patch("chronos.chronos2.pipeline.Chronos2Pipeline.fit") as mocked_pipeline_fit:
            model.fit(data)
            inputs = mocked_pipeline_fit.call_args.kwargs["inputs"]

            for input_dict in inputs:
                if disable_past:
                    for col in covariate_metadata.past_covariates:
                        assert col not in input_dict.get("past_covariates", {})
                if disable_known:
                    assert "future_covariates" not in input_dict

    @pytest.mark.parametrize(
        "disable_past,disable_known",
        [(True, False), (False, True), (True, True)],
    )
    def test_when_covariates_disabled_then_not_used_during_predict(
        self, df_with_covariates, disable_past, disable_known
    ):
        data, covariate_metadata = df_with_covariates
        prediction_length = 5
        past_data = data.slice_by_timestep(None, -prediction_length)
        future_data = data.slice_by_timestep(-prediction_length, None)
        known_covariates = future_data.drop(columns=["target"])

        model = Chronos2Model(
            prediction_length=prediction_length,
            hyperparameters={
                "model_path": CHRONOS2_MODEL_PATH,
                "disable_past_covariates": disable_past,
                "disable_known_covariates": disable_known,
            },
            covariate_metadata=covariate_metadata,
        )
        model.fit(past_data)

        with mock.patch("chronos.chronos2.pipeline.Chronos2Pipeline.predict_df") as mocked_predict_df:
            try:
                model.predict(past_data, known_covariates=known_covariates)
            except ValueError:
                pass

            call_kwargs = mocked_predict_df.call_args.kwargs
            df_columns = set(call_kwargs["df"].columns)

            if disable_past:
                for cov in covariate_metadata.past_covariates:
                    assert cov not in df_columns
            if disable_known:
                assert call_kwargs["future_df"] is None
                for cov in covariate_metadata.known_covariates:
                    assert cov not in df_columns
