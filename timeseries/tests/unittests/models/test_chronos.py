from typing import Optional

import numpy as np
import pytest
import torch

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
        "model_path": "tiny",
    },
]


@pytest.fixture(
    scope="module",
    params=[
        {
            "optimization_strategy": "onnx",
        },
        *HYPERPARAMETER_DICTS,
    ],
)
def default_chronos_tiny_model(request) -> ChronosModel:
    model = ChronosModel(
        hyperparameters={
            "model_path": "amazon/chronos-t5-tiny",
            "num_samples": 5,
            "device": "cpu",
            "context_length": 16,
            **request.param,
        },
    )
    model.fit(train_data=None)
    return model


@pytest.fixture(scope="module", params=HYPERPARAMETER_DICTS)
def default_chronos_tiny_model_gpu(request) -> Optional[ChronosModel]:
    if not GPU_AVAILABLE:
        return None

    model = ChronosModel(
        hyperparameters={
            "model_path": "amazon/chronos-t5-tiny",
            "device": "cuda",
            **request.param,
        },
    )
    model.fit(train_data=None)

    # load pipeline in fixture to speed up tests
    model.load_model_pipeline()

    return model


@pytest.mark.parametrize("data", DATASETS)
def test_when_on_cpu_then_chronos_model_can_score_and_cache_oof(data, default_chronos_tiny_model):
    default_chronos_tiny_model.score_and_cache_oof(data)
    assert default_chronos_tiny_model._oof_predictions is not None


@pytest.mark.parametrize("data", DATASETS)
def test_when_on_cpu_then_chronos_model_can_infer(data, default_chronos_tiny_model):
    predictions = default_chronos_tiny_model.predict(data)
    assert all(
        predictions.index.get_level_values("item_id").unique() == data.index.get_level_values("item_id").unique()
    )


def test_given_nan_features_when_on_cpu_then_chronos_model_inferences_not_nan(default_chronos_tiny_model):
    data = get_data_frame_with_variable_lengths({"A": 20, "B": 12}, covariates_names=["cov1", "cov2", "cov3"])
    data[["cov1", "cov2", "cov3"]] = np.nan

    predictions = default_chronos_tiny_model.predict(data)
    assert all(
        predictions.index.get_level_values("item_id").unique() == data.index.get_level_values("item_id").unique()
    )
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
    assert all(
        predictions.index.get_level_values("item_id").unique() == data.index.get_level_values("item_id").unique()
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Requires GPU")
def test_given_nan_features_when_on_gpu_then_chronos_model_inferences_not_nan(default_chronos_tiny_model_gpu):
    data = get_data_frame_with_variable_lengths({"A": 20, "B": 12}, covariates_names=["cov1", "cov2", "cov3"])
    data[["cov1", "cov2", "cov3"]] = np.nan

    predictions = default_chronos_tiny_model_gpu.predict(data)
    assert all(
        predictions.index.get_level_values("item_id").unique() == data.index.get_level_values("item_id").unique()
    )
    assert not any(predictions["mean"].isna())


@pytest.mark.parametrize("batch_size", [6, 12])
def test_when_batch_size_provided_then_batch_size_used_to_infer(batch_size):
    data = get_data_frame_with_item_index(list(range(20)))
    model = ChronosModel(
        hyperparameters={
            "model_path": "amazon/chronos-t5-tiny",
            "device": "cpu",
            "batch_size": batch_size,
        },
    )
    model.fit(train_data=None)
    loader = model.get_inference_data_loader(data)
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
    assert loaded_model.model_pipeline is not None

    predictions = default_chronos_tiny_model.predict(data)
    assert all(
        predictions.index.get_level_values("item_id").unique() == data.index.get_level_values("item_id").unique()
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Requires GPU")
@pytest.mark.parametrize("data", DATASETS)
def test_when_gpu_models_saved_then_models_can_be_loaded_and_inferred(data, default_chronos_tiny_model_gpu):
    default_chronos_tiny_model_gpu.save()

    loaded_model = default_chronos_tiny_model_gpu.__class__.load(path=default_chronos_tiny_model_gpu.path)
    assert loaded_model.model_pipeline is not None

    predictions = default_chronos_tiny_model_gpu.predict(data)
    assert all(
        predictions.index.get_level_values("item_id").unique() == data.index.get_level_values("item_id").unique()
    )


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ],
)
def test_when_torch_dtype_provided_then_parameters_loaded_in_torch_dtype(dtype):
    model = ChronosModel(
        hyperparameters={
            "model_path": "amazon/chronos-t5-tiny",
            "device": "cpu",
            "torch_dtype": dtype,
        },
    )
    model.fit(train_data=None)
    model.load_model_pipeline()

    embedding_matrix = next(iter(model.model_pipeline.model.model.shared.parameters()))
    assert embedding_matrix.dtype is dtype
