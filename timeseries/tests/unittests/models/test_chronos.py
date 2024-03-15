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
HYPERPARAMETER_SETS = []
GPU_AVAILABLE = torch.cuda.is_available()


@pytest.fixture(scope="module")
def default_chronos_tiny_model() -> ChronosModel:
    model = ChronosModel(
        hyperparameters={
            "model_path": "amazon/chronos-t5-tiny",
            "device": "cpu",
        },
    )
    model.fit(train_data=None)
    return model


@pytest.fixture(scope="module")
def default_chronos_tiny_model_gpu() -> Optional[ChronosModel]:
    if not GPU_AVAILABLE:
        return None

    model = ChronosModel(
        hyperparameters={
            "model_path": "amazon/chronos-t5-tiny",
            "device": "cuda",
        },
    )
    model.fit(train_data=None)
    return model


@pytest.mark.parametrize("data", DATASETS)
def test_when_on_cpu_then_chronos_model_can_score_and_cache_oof(data, default_chronos_tiny_model):
    default_chronos_tiny_model.score_and_cache_oof(data)
    if not default_chronos_tiny_model.skip_validation:
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
    if not default_chronos_tiny_model_gpu.skip_validation:
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

    loaded_model = default_chronos_tiny_model_gpu.__class__.load(path=default_chronos_tiny_model.path)
    assert loaded_model.model_pipeline is not None

    predictions = default_chronos_tiny_model_gpu.predict(data)
    assert all(
        predictions.index.get_level_values("item_id").unique() == data.index.get_level_values("item_id").unique()
    )
