import copy
from typing import Optional
from unittest import mock

import numpy as np
import pytest
import torch

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries import TimeSeriesPredictor
from autogluon.timeseries.models import ChronosModel

from ...common import (
    DATAFRAME_WITH_COVARIATES,
    DATAFRAME_WITH_STATIC,
    DUMMY_TS_DATAFRAME,
    get_data_frame_with_item_index,
    get_data_frame_with_variable_lengths,
)
from .conftest import CHRONOS_BOLT_TEST_MODEL_PATH

DATASETS = [DUMMY_TS_DATAFRAME, DATAFRAME_WITH_STATIC, DATAFRAME_WITH_COVARIATES]
GPU_AVAILABLE = torch.cuda.is_available()
HYPERPARAMETER_DICTS = [
    {
        "batch_size": 4,
    },
    {
        "context_length": 64,
    },
    {
        "context_length": None,
    },
    {
        "fine_tune": True,
        "fine_tune_steps": 2,
    },
    {
        "fine_tune": True,
        "fine_tune_steps": 2,
        "context_length": 64,
    },
]


def chronos_bolt_model(*args, hyperparameters=None, **kwargs):
    hyperparameters = copy.deepcopy(hyperparameters or {})
    hyperparameters |= {"model_path": CHRONOS_BOLT_TEST_MODEL_PATH}
    kwargs["hyperparameters"] = hyperparameters
    return ChronosModel(*args, **kwargs)


def chronos_with_finetuning(*args, hyperparameters=None, **kwargs):
    hyperparameters = copy.deepcopy(hyperparameters or {})
    hyperparameters |= {"fine_tune": True, "fine_tune_steps": 10}
    kwargs["hyperparameters"] = hyperparameters
    return ChronosModel(*args, **kwargs)


def chronos_bolt_model_with_finetuning(*args, hyperparameters=None, **kwargs):
    hyperparameters = copy.deepcopy(hyperparameters or {})
    hyperparameters |= {"model_path": CHRONOS_BOLT_TEST_MODEL_PATH, "fine_tune": True, "fine_tune_steps": 10}
    kwargs["hyperparameters"] = hyperparameters
    return ChronosModel(*args, **kwargs)


ZERO_SHOT_MODELS = [ChronosModel, chronos_bolt_model]
TESTABLE_MODELS = [ChronosModel, chronos_with_finetuning, chronos_bolt_model, chronos_bolt_model_with_finetuning]


@pytest.fixture(
    scope="module",
    params=HYPERPARAMETER_DICTS,
)
def default_chronos_tiny_model(request, chronos_model_path) -> ChronosModel:
    model = ChronosModel(
        hyperparameters={
            "model_path": chronos_model_path,
            "num_samples": 3,
            "context_length": 16,
            "device": "cpu",
            **request.param,
        },
    )
    return model


@pytest.fixture(scope="module", params=HYPERPARAMETER_DICTS)
def default_chronos_tiny_model_gpu(request, chronos_model_path) -> Optional[ChronosModel]:
    if not GPU_AVAILABLE:
        return None

    model = ChronosModel(
        hyperparameters={
            "model_path": chronos_model_path,
            "device": "cuda",
            **request.param,
        },
    )
    return model


@pytest.mark.parametrize("data", DATASETS)
def test_when_on_cpu_then_chronos_model_can_score_and_cache_oof(data, default_chronos_tiny_model):
    default_chronos_tiny_model.fit(train_data=data)
    default_chronos_tiny_model.score_and_cache_oof(data)
    assert default_chronos_tiny_model._oof_predictions is not None


@pytest.mark.parametrize("data", DATASETS)
def test_when_on_cpu_then_chronos_model_can_infer(data, default_chronos_tiny_model):
    default_chronos_tiny_model.fit(train_data=data)
    predictions = default_chronos_tiny_model.predict(data)
    assert all(predictions.item_ids == data.item_ids)


@pytest.mark.parametrize("data", DATASETS)
def test_when_on_cpu_and_model_requested_from_hf_then_chronos_model_can_infer(data):
    model = ChronosModel(
        hyperparameters={"model_path": "tiny", "device": "cpu"},
    )
    model.fit(train_data=None)
    predictions = model.predict(data)

    assert all(predictions.item_ids == data.item_ids)


def test_given_nan_features_when_on_cpu_then_chronos_model_inferences_not_nan(default_chronos_tiny_model):
    data = get_data_frame_with_variable_lengths({"A": 20, "B": 12}, covariates_names=["cov1", "cov2", "cov3"])
    data[["cov1", "cov2", "cov3"]] = np.nan
    default_chronos_tiny_model.fit(train_data=data)
    predictions = default_chronos_tiny_model.predict(data)
    assert all(predictions.item_ids == data.item_ids)
    assert not any(predictions["mean"].isna())


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Requires GPU")
@pytest.mark.parametrize("data", DATASETS)
def test_when_on_gpu_then_chronos_model_can_score_and_cache_oof(data, default_chronos_tiny_model_gpu):
    default_chronos_tiny_model_gpu.fit(train_data=data)
    default_chronos_tiny_model_gpu.score_and_cache_oof(data)
    assert default_chronos_tiny_model_gpu._oof_predictions is not None


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Requires GPU")
@pytest.mark.parametrize("data", DATASETS)
def test_when_on_gpu_then_chronos_model_can_infer(data, default_chronos_tiny_model_gpu):
    default_chronos_tiny_model_gpu.fit(train_data=data)
    default_chronos_tiny_model_gpu.fit(train_data=data)
    predictions = default_chronos_tiny_model_gpu.predict(data)
    assert all(predictions.item_ids == data.item_ids)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Requires GPU")
def test_given_nan_features_when_on_gpu_then_chronos_model_inferences_not_nan(default_chronos_tiny_model_gpu):
    data = get_data_frame_with_variable_lengths({"A": 20, "B": 12}, covariates_names=["cov1", "cov2", "cov3"])
    data[["cov1", "cov2", "cov3"]] = np.nan

    default_chronos_tiny_model_gpu.fit(train_data=data)
    predictions = default_chronos_tiny_model_gpu.predict(data)
    assert all(predictions.item_ids == data.item_ids)
    assert not any(predictions["mean"].isna())


@pytest.mark.parametrize("batch_size", [6, 12])
def test_when_batch_size_provided_then_batch_size_used_to_infer(batch_size, chronos_model_path):
    data = get_data_frame_with_item_index(list(range(20)))
    model = ChronosModel(
        hyperparameters={
            "model_path": chronos_model_path,
            "device": "cpu",
            "batch_size": batch_size,
            "context_length": 16,
        },
    )
    model.fit(train_data=None)
    loader = model._get_inference_data_loader(data, context_length=16)
    batch = next(iter(loader))

    assert batch.shape[0] == batch_size


@pytest.mark.parametrize("data", DATASETS)
def test_when_cpu_models_saved_then_models_can_be_loaded_and_inferred(data, default_chronos_tiny_model):
    default_chronos_tiny_model.fit(train_data=data)
    default_chronos_tiny_model.save()

    loaded_model = default_chronos_tiny_model.__class__.load(path=default_chronos_tiny_model.path)

    predictions = loaded_model.predict(data)
    assert all(predictions.item_ids == data.item_ids)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Requires GPU")
@pytest.mark.parametrize("data", DATASETS)
def test_when_gpu_models_saved_then_models_can_be_loaded_and_inferred(data, default_chronos_tiny_model_gpu):
    default_chronos_tiny_model_gpu.fit(train_data=data)
    default_chronos_tiny_model_gpu.save()

    loaded_model = default_chronos_tiny_model_gpu.__class__.load(path=default_chronos_tiny_model_gpu.path)

    predictions = loaded_model.predict(data)
    assert all(predictions.item_ids == data.item_ids)


@pytest.mark.parametrize(
    "data_length, expected_context_length", [(5, 5), (7, 7), (5000, ChronosModel.maximum_context_length)]
)
def test_when_context_length_not_provided_then_context_length_set_to_dataset_length(
    chronos_model_path, data_length, expected_context_length
):
    data = get_data_frame_with_item_index(list(range(3)), data_length=data_length)
    model = ChronosModel(hyperparameters={"model_path": chronos_model_path})
    model.fit(train_data=None)
    model.persist()  # persist so that we can patch the predict method

    with mock.patch.object(model.model_pipeline, "predict_quantiles") as patch_predict_quantiles:
        try:
            model.predict(data)
        except ValueError:
            pass

        batch = patch_predict_quantiles.call_args.args[0]

    assert batch.shape[-1] == expected_context_length


@pytest.mark.parametrize(
    "init_context_length, data_length, expected_context_length",
    [
        (64, 5, 64),
        (32, 7, 32),
        (32, 64, 32),
        (10000, 30, ChronosModel.maximum_context_length),
        (10000, 5000, ChronosModel.maximum_context_length),
    ],
)
def test_when_context_length_provided_then_context_length_set_to_capped_init_context_length(
    chronos_model_path, init_context_length, data_length, expected_context_length
):
    data = get_data_frame_with_item_index(list(range(3)), data_length=data_length)
    model = ChronosModel(hyperparameters={"model_path": chronos_model_path, "context_length": init_context_length})
    model.fit(train_data=None)
    model.persist()  # persist so that we can patch the predict method

    with mock.patch.object(model.model_pipeline, "predict_quantiles") as patch_predict_quantiles:
        try:
            model.predict(data)
        except ValueError:
            pass

        batch = patch_predict_quantiles.call_args.args[0]

    assert batch.shape[-1] == expected_context_length


@pytest.mark.parametrize(
    "longest_data_length, expected_context_length", [(5, 5), (7, 7), (5000, ChronosModel.maximum_context_length)]
)
def test_given_variable_length_data_when_context_length_not_provided_then_context_length_set_to_max_data_length(
    chronos_model_path, longest_data_length, expected_context_length
):
    data = get_data_frame_with_variable_lengths({"A": 3, "B": 3, "C": longest_data_length})
    model = ChronosModel(hyperparameters={"model_path": chronos_model_path})
    model.fit(train_data=None)
    model.persist()  # persist so that we can patch the predict method

    with mock.patch.object(model.model_pipeline, "predict_quantiles") as patch_predict_quantiles:
        try:
            model.predict(data)
        except ValueError:
            pass

        batch = patch_predict_quantiles.call_args.args[0]

    assert batch.shape[-1] == expected_context_length


DTYPE_TEST_CASES = [  # dtype_arg, expected_dtype
    (torch.float16, torch.float16),
    (torch.bfloat16, torch.bfloat16),
    (torch.float32, torch.float32),
    (torch.float64, torch.float64),
    ("bfloat16", torch.bfloat16),
    ("float32", torch.float32),
    ("float64", torch.float64),
]


@pytest.mark.parametrize("dtype_arg, expected_dtype", DTYPE_TEST_CASES)
def test_when_torch_dtype_provided_then_parameters_loaded_in_torch_dtype(
    chronos_model_path, dtype_arg, expected_dtype
):
    model = ChronosModel(
        hyperparameters={
            "model_path": chronos_model_path,
            "device": "cpu",
            "torch_dtype": dtype_arg,
        },
    )
    model.fit(train_data=None)
    model.load_model_pipeline()

    parameter = next(iter(model.model_pipeline.model.parameters()))
    assert parameter.dtype is expected_dtype


@pytest.mark.parametrize("dtype_arg, expected_dtype", DTYPE_TEST_CASES)
def test_when_torch_dtype_provided_and_model_persisted_then_parameters_loaded_in_torch_dtype(
    chronos_model_path, dtype_arg, expected_dtype
):
    model = ChronosModel(
        hyperparameters={
            "model_path": chronos_model_path,
            "device": "cpu",
            "torch_dtype": dtype_arg,
        },
    )
    model.persist()

    parameter = next(iter(model.model_pipeline.model.parameters()))
    assert parameter.dtype is expected_dtype


def test_when_model_persisted_then_model_pipeline_can_infer(chronos_model_path):
    model = ChronosModel(
        hyperparameters={
            "model_path": chronos_model_path,
            "device": "cpu",
        },
    )
    model.persist()
    assert model.model_pipeline.predict(torch.tensor([[1, 2, 3]])) is not None


def test_when_model_not_persisted_only_fit_then_model_pipeline_is_none(chronos_model_path):
    model = ChronosModel(
        hyperparameters={
            "model_path": chronos_model_path,
            "device": "cpu",
        },
    )
    model._fit(DUMMY_TS_DATAFRAME)
    assert model._model_pipeline is None


def test_when_model_saved_loaded_and_persisted_then_model_pipeline_can_infer(chronos_model_path):
    model = ChronosModel(
        hyperparameters={
            "model_path": chronos_model_path,
            "device": "cpu",
        },
    )
    path = model.save()
    model = ChronosModel.load(path)

    model.persist()
    assert model.model_pipeline.predict(torch.tensor([[1, 2, 3]])) is not None


def test_when_chronos_fit_in_standalone_through_predictor_and_persist_called_then_chronos_pipeline_is_persisted(
    chronos_model_path,
    temp_model_path,
):
    predictor = TimeSeriesPredictor(path=temp_model_path).fit(
        DUMMY_TS_DATAFRAME,
        skip_model_selection=True,
        hyperparameters={"Chronos": {"model_path": chronos_model_path}},
        enable_ensemble=False,
    )
    predictor.persist()
    name, model = next(iter(predictor._learner.trainer.models.items()))
    assert "Chronos" in name
    assert model.model_pipeline is not None


def test_when_chronos_fit_with_validation_through_predictor_and_persist_called_then_chronos_pipeline_is_persisted(
    chronos_model_path,
    temp_model_path,
):
    predictor = TimeSeriesPredictor(path=temp_model_path).fit(
        DUMMY_TS_DATAFRAME,
        hyperparameters={"Chronos": {"model_path": chronos_model_path}},
        enable_ensemble=False,
    )
    predictor.persist()
    name, model = next(iter(predictor._learner.trainer.models.items()))
    assert "Chronos" in name

    # model now wrapped in MultiWindowModel
    assert model.most_recent_model.model_pipeline is not None


@pytest.mark.parametrize("data_loader_num_workers", [0, 1, 2])
def test_when_chronos_scores_oof_and_time_limit_is_exceeded_then_exception_is_raised(
    chronos_model_path, temp_model_path, data_loader_num_workers
):
    data = get_data_frame_with_item_index(item_list=list(range(1000)), data_length=50)
    model = ChronosModel(
        prediction_length=20,
        path=temp_model_path,
        hyperparameters={"model_path": chronos_model_path, "data_loader_num_workers": data_loader_num_workers},
    )
    model.fit(data)
    with pytest.raises(TimeLimitExceeded):
        model.score_and_cache_oof(data, time_limit=0.1)


def test_when_eval_during_fine_tune_is_false_then_evaluation_is_turned_off(chronos_model_path):
    model = ChronosModel(
        hyperparameters={
            "model_path": chronos_model_path,
            "device": "cpu",
            "fine_tune": True,
            "eval_during_fine_tune": False,
        },
    )

    with mock.patch("transformers.trainer.TrainingArguments.__init__") as training_args:
        try:
            model.fit(DUMMY_TS_DATAFRAME)
        except TypeError:
            pass

        assert training_args.call_args.kwargs["evaluation_strategy"] == "no"
        assert training_args.call_args.kwargs["eval_steps"] is None
        assert not training_args.call_args.kwargs["load_best_model_at_end"]
        assert training_args.call_args.kwargs["metric_for_best_model"] is None


@pytest.mark.parametrize("max_items", [3, 20, None])
def test_fine_tune_eval_max_items_is_used(chronos_model_path, max_items):
    model = ChronosModel(
        hyperparameters={
            "model_path": chronos_model_path,
            "device": "cpu",
            "fine_tune": True,
            "fine_tune_eval_max_items": max_items,
        },
    )
    expected_max_items = (
        min(max_items, DUMMY_TS_DATAFRAME.num_items) if max_items is not None else DUMMY_TS_DATAFRAME.num_items
    )

    with mock.patch(
        "autogluon.timeseries.models.chronos.pipeline.utils.ChronosFineTuningDataset.__init__"
    ) as chronos_ft_dataset:
        chronos_ft_dataset.side_effect = [None, None]

        try:
            model.fit(DUMMY_TS_DATAFRAME, val_data=DUMMY_TS_DATAFRAME)
        except AttributeError:
            pass

        val_data_subset = chronos_ft_dataset.call_args_list[1].kwargs["target_df"]

        assert val_data_subset.num_items == expected_max_items


@pytest.mark.parametrize("shuffle_buffer_size", [20, None])
def test_fine_tune_shuffle_buffer_size_is_used(chronos_model_path, shuffle_buffer_size):
    model = ChronosModel(
        hyperparameters={
            "model_path": chronos_model_path,
            "device": "cpu",
            "fine_tune": True,
            "fine_tune_shuffle_buffer_size": shuffle_buffer_size,
        },
    )

    with mock.patch(
        "autogluon.timeseries.models.chronos.pipeline.utils.ChronosFineTuningDataset.shuffle"
    ) as chronos_ft_dataset_shuffle:
        try:
            model.fit(DUMMY_TS_DATAFRAME)
        except ValueError:
            pass

        assert chronos_ft_dataset_shuffle.call_args.args[0] == shuffle_buffer_size
