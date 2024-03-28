from typing import Optional

import numpy as np
import pytest
import torch

from autogluon.timeseries import TimeSeriesPredictor
from autogluon.timeseries.models import ChronosModel
from autogluon.timeseries.models.chronos.model import ChronosInferenceDataset

from ..common import (
    DATAFRAME_WITH_COVARIATES,
    DATAFRAME_WITH_STATIC,
    DUMMY_TS_DATAFRAME,
    get_data_frame_with_item_index,
    get_data_frame_with_variable_lengths,
)

DATASETS = [DUMMY_TS_DATAFRAME, DATAFRAME_WITH_STATIC, DATAFRAME_WITH_COVARIATES]
TESTABLE_MODELS = [ChronosModel]
GPU_AVAILABLE = torch.cuda.is_available()
HYPERPARAMETER_DICTS = [
    {
        "batch_size": 32,
    },
    {
        "batch_size": 4,
    },
    {
        "num_samples": 10,
    },
    {
        "context_length": 64,
    },
    {
        "context_length": None,
    },
]


@pytest.fixture(
    scope="module",
    params=[
        {
            "optimization_strategy": "onnx",
        },
        {
            "optimization_strategy": "openvino",
        },
        *HYPERPARAMETER_DICTS,
    ],
)
def default_chronos_tiny_model(request, hf_model_path) -> ChronosModel:
    model = ChronosModel(
        hyperparameters={
            "model_path": hf_model_path,
            "num_samples": 3,
            "context_length": 16,
            "device": "cpu",
            **request.param,
        },
    )
    model.fit(train_data=None)
    return model


@pytest.fixture(scope="module", params=HYPERPARAMETER_DICTS)
def default_chronos_tiny_model_gpu(request, hf_model_path) -> Optional[ChronosModel]:
    if not GPU_AVAILABLE:
        return None

    model = ChronosModel(
        hyperparameters={
            "model_path": hf_model_path,
            "device": "cuda",
            **request.param,
        },
    )
    model.fit(train_data=None)
    return model


@pytest.mark.parametrize("data", DATASETS)
def test_when_on_cpu_then_chronos_model_can_score_and_cache_oof(data, default_chronos_tiny_model):
    default_chronos_tiny_model.score_and_cache_oof(data)
    assert default_chronos_tiny_model._oof_predictions is not None


@pytest.mark.parametrize("data", DATASETS)
def test_when_on_cpu_then_chronos_model_can_infer(data, default_chronos_tiny_model):
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

    predictions = default_chronos_tiny_model.predict(data)
    assert all(predictions.item_ids == data.item_ids)
    assert not any(predictions["mean"].isna())


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Requires GPU")
@pytest.mark.parametrize("data", DATASETS)
def test_when_on_gpu_then_chronos_model_can_score_and_cache_oof(data, default_chronos_tiny_model_gpu):
    default_chronos_tiny_model_gpu.score_and_cache_oof(data)
    assert default_chronos_tiny_model_gpu._oof_predictions is not None


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Requires GPU")
@pytest.mark.parametrize("data", DATASETS)
def test_when_on_gpu_then_chronos_model_can_infer(data, default_chronos_tiny_model_gpu):
    predictions = default_chronos_tiny_model_gpu.predict(data)
    assert all(predictions.item_ids == data.item_ids)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Requires GPU")
def test_given_nan_features_when_on_gpu_then_chronos_model_inferences_not_nan(default_chronos_tiny_model_gpu):
    data = get_data_frame_with_variable_lengths({"A": 20, "B": 12}, covariates_names=["cov1", "cov2", "cov3"])
    data[["cov1", "cov2", "cov3"]] = np.nan

    predictions = default_chronos_tiny_model_gpu.predict(data)
    assert all(predictions.item_ids == data.item_ids)
    assert not any(predictions["mean"].isna())


@pytest.mark.parametrize("batch_size", [6, 12])
def test_when_batch_size_provided_then_batch_size_used_to_infer(batch_size, hf_model_path):
    data = get_data_frame_with_item_index(list(range(20)))
    model = ChronosModel(
        hyperparameters={
            "model_path": hf_model_path,
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
@pytest.mark.parametrize("context_length", [5, 10, 20])
def test_when_context_length_provided_then_inference_dataset_context_length_used(data, context_length):
    inference_dataset = ChronosInferenceDataset(data, context_length=context_length)
    item = inference_dataset[0]

    assert item.shape[-1] == context_length


@pytest.mark.parametrize("context_length", [5, 10, 20])
def test_when_context_length_provided_then_padding_correct(context_length):
    data = get_data_frame_with_item_index(list(range(20)), data_length=5)
    inference_dataset = ChronosInferenceDataset(data, context_length=context_length)
    item = inference_dataset[0]

    assert np.sum(np.isnan(item)) == context_length - 5
    assert not np.isnan(item[-1])  # padding left


@pytest.mark.parametrize(
    "item_id_to_length, expected_indptr",
    [
        ({"A": 20, "B": 12}, [0, 20, 32]),
        ({"A": 20, "B": 12, "C": 1}, [0, 20, 32, 33]),
        ({"A": 20}, [0, 20]),
        ({"A": 1}, [0, 1]),
        ({"B": 10, "A": 10}, [0, 10, 20]),
    ],
)
def test_when_inference_dataset_initialized_then_indptr_set_correctly(item_id_to_length, expected_indptr):
    dataset = get_data_frame_with_variable_lengths(item_id_to_length)
    inference_dataset = ChronosInferenceDataset(dataset, context_length=5)

    assert inference_dataset.indptr.tolist() == expected_indptr


@pytest.mark.parametrize("data", DATASETS)
def test_when_cpu_models_saved_then_models_can_be_loaded_and_inferred(data, default_chronos_tiny_model):
    default_chronos_tiny_model.save()

    loaded_model = default_chronos_tiny_model.__class__.load(path=default_chronos_tiny_model.path)

    predictions = loaded_model.predict(data)
    assert all(predictions.item_ids == data.item_ids)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Requires GPU")
@pytest.mark.parametrize("data", DATASETS)
def test_when_gpu_models_saved_then_models_can_be_loaded_and_inferred(data, default_chronos_tiny_model_gpu):
    default_chronos_tiny_model_gpu.save()

    loaded_model = default_chronos_tiny_model_gpu.__class__.load(path=default_chronos_tiny_model_gpu.path)

    predictions = loaded_model.predict(data)
    assert all(predictions.item_ids == data.item_ids)


@pytest.mark.parametrize(
    "data_length, expected_context_length", [(5, 5), (7, 7), (1000, ChronosModel.maximum_context_length)]
)
def test_when_context_length_not_provided_then_context_length_set_to_dataset_length(
    hf_model_path, data_length, expected_context_length
):
    data = get_data_frame_with_item_index(list(range(3)), data_length=data_length)
    model = ChronosModel(hyperparameters={"model_path": hf_model_path})
    model.fit(train_data=None)
    model.predict(data)

    assert model.model_pipeline.model.config.context_length == expected_context_length


@pytest.mark.parametrize(
    "init_context_length, data_length, expected_context_length",
    [
        (64, 5, 64),
        (32, 7, 32),
        (32, 64, 32),
        (10000, 30, ChronosModel.maximum_context_length),
        (10000, 1000, ChronosModel.maximum_context_length),
    ],
)
def test_when_context_length_provided_then_context_length_set_to_capped_init_context_length(
    hf_model_path, init_context_length, data_length, expected_context_length
):
    data = get_data_frame_with_item_index(list(range(3)), data_length=data_length)
    model = ChronosModel(hyperparameters={"model_path": hf_model_path, "context_length": init_context_length})
    model.fit(train_data=None)
    model.predict(data)

    assert model.model_pipeline.model.config.context_length == expected_context_length


@pytest.mark.parametrize(
    "longest_data_length, expected_context_length", [(5, 5), (7, 7), (1000, ChronosModel.maximum_context_length)]
)
def test_given_variable_length_data_when_context_length_not_provided_then_context_length_set_to_max_data_length(
    hf_model_path, longest_data_length, expected_context_length
):
    data = get_data_frame_with_variable_lengths({"A": 3, "B": 3, "C": longest_data_length})
    model = ChronosModel(hyperparameters={"model_path": hf_model_path})
    model.fit(train_data=None)
    model.predict(data)

    assert model.model_pipeline.model.config.context_length == expected_context_length


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
def test_when_torch_dtype_provided_then_parameters_loaded_in_torch_dtype(hf_model_path, dtype_arg, expected_dtype):
    model = ChronosModel(
        hyperparameters={
            "model_path": hf_model_path,
            "device": "cpu",
            "torch_dtype": dtype_arg,
        },
    )
    model.fit(train_data=None)
    model.load_model_pipeline()

    embedding_matrix = next(iter(model.model_pipeline.model.model.shared.parameters()))
    assert embedding_matrix.dtype is expected_dtype


@pytest.mark.parametrize("dtype_arg, expected_dtype", DTYPE_TEST_CASES)
def test_when_torch_dtype_provided_and_model_persisted_then_parameters_loaded_in_torch_dtype(
    hf_model_path, dtype_arg, expected_dtype
):
    model = ChronosModel(
        hyperparameters={
            "model_path": hf_model_path,
            "device": "cpu",
            "torch_dtype": dtype_arg,
        },
    )
    model.persist()

    embedding_matrix = next(iter(model.model_pipeline.model.model.shared.parameters()))
    assert embedding_matrix.dtype is expected_dtype


def test_when_model_persisted_then_model_pipeline_can_infer(hf_model_path):
    model = ChronosModel(
        hyperparameters={
            "model_path": hf_model_path,
            "device": "cpu",
        },
    )
    model.persist()
    assert model.model_pipeline.predict(torch.tensor([[1, 2, 3]])) is not None


def test_when_model_not_persisted_only_fit_then_model_pipeline_is_none(hf_model_path):
    model = ChronosModel(
        hyperparameters={
            "model_path": hf_model_path,
            "device": "cpu",
        },
    )
    model._fit(DUMMY_TS_DATAFRAME)
    assert model.model_pipeline is None


def test_when_model_saved_loaded_and_persisted_then_model_pipeline_can_infer(hf_model_path):
    model = ChronosModel(
        hyperparameters={
            "model_path": hf_model_path,
            "device": "cpu",
        },
    )
    path = model.save()
    model = ChronosModel.load(path)

    model.persist()
    assert model.model_pipeline.predict(torch.tensor([[1, 2, 3]])) is not None


def test_when_chronos_fit_in_standalone_through_predictor_and_persist_called_then_chronos_pipeline_is_persisted(
    hf_model_path,
    temp_model_path,
):
    predictor = TimeSeriesPredictor(path=temp_model_path).fit(
        DUMMY_TS_DATAFRAME,
        skip_model_selection=True,
        hyperparameters={"Chronos": {"model_path": hf_model_path}},
        enable_ensemble=False,
    )
    predictor.persist()
    name, model = next(iter(predictor._learner.trainer.models.items()))
    assert "Chronos" in name
    assert model.model_pipeline is not None


def test_when_chronos_fit_with_validation_through_predictor_and_persist_called_then_chronos_pipeline_is_persisted(
    hf_model_path,
    temp_model_path,
):
    predictor = TimeSeriesPredictor(path=temp_model_path).fit(
        DUMMY_TS_DATAFRAME,
        hyperparameters={"Chronos": {"model_path": hf_model_path}},
        enable_ensemble=False,
    )
    predictor.persist()
    name, model = next(iter(predictor._learner.trainer.models.items()))
    assert "Chronos" in name

    # model now wrapped in MultiWindowModel
    assert model.most_recent_model.model_pipeline is not None
