import os
import shutil
from unittest import mock

import pytest

from autogluon.timeseries.models.chronos import Chronos2Model

from ...common import DATAFRAME_WITH_COVARIATES, DUMMY_TS_DATAFRAME
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
                    print(known_covariates.columns, input_dict.keys())
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
