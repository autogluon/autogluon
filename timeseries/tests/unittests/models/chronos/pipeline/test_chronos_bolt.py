import pytest
import torch

from autogluon.timeseries.models.chronos.pipeline import BaseChronosPipeline
from autogluon.timeseries.models.chronos.pipeline.chronos_bolt import InstanceNorm, Patch

from ..test_model import CHRONOS_BOLT_MODEL_PATH


def validate_tensor(input: torch.Tensor, shape: tuple[int, ...]) -> None:
    assert isinstance(input, torch.Tensor)
    assert input.shape == shape


@pytest.fixture(scope="module", params=[torch.float32, torch.bfloat16])
def pipeline(request):
    """Fixture to create a Chronos-Bolt pipeline with a given torch dtype"""
    return BaseChronosPipeline.from_pretrained(
        CHRONOS_BOLT_MODEL_PATH,
        device_map="cpu",
        torch_dtype=request.param,
    )


@pytest.fixture(scope="module", params=["batch-tensor", "batch-list", "single-sequence"])
def context(request):
    """Dummy context with batch size of 4 and sequence length of 16, along with a batch size
    associated with the context"""
    context = 10 * torch.rand(size=(4, 16)) + 10
    if request.param == "batch-list":
        return list(context), 4
    elif request.param == "batch-tensor":
        return context, 4
    elif request.param == "single-sequence":
        return context[0], 1


def test_given_context_pipeline_can_predict(pipeline, context):
    batch, batch_size = context
    quantiles = pipeline.predict(batch, prediction_length=3)
    validate_tensor(quantiles, (batch_size, len(pipeline.quantiles), 3))


def test_given_long_prediction_length_and_limit_prediction_length_true_pipeline_predict_fails(pipeline, context):
    batch, _ = context
    with pytest.raises(ValueError):
        _ = pipeline.predict(batch, prediction_length=65, limit_prediction_length=True)


def test_given_long_prediction_length_and_limit_prediction_length_false_pipeline_can_predict(pipeline, context):
    batch, batch_size = context
    quantiles = pipeline.predict(batch, prediction_length=65, limit_prediction_length=False)
    validate_tensor(quantiles, (batch_size, len(pipeline.quantiles), 65))


def test_given_even_data_patch_operator_output_is_correct():
    batch_size = 17
    patch_len = 16

    patch = Patch(patch_len, patch_len)

    batch = torch.stack([torch.arange(512)] * batch_size) + torch.arange(batch_size)[:, None]
    output = patch(batch)

    assert output.shape == (batch_size, 512 // patch_len, patch_len)

    assert torch.allclose(
        output[:, 0],
        torch.stack([torch.arange(patch_len)] * batch_size) + torch.arange(batch_size)[:, None],
        atol=1e-5,
    )
    assert torch.allclose(
        output[:, 1],
        torch.stack([torch.arange(patch_len, 2 * patch_len)] * batch_size) + torch.arange(batch_size)[:, None],
        atol=1e-5,
    )
    assert not torch.isnan(output).any()


def test_given_even_data_and_strides_patch_operator_output_is_correct():
    batch_size = 17
    patch_len, patch_stride = 16, 8

    patch = Patch(patch_len, patch_stride)

    offset = torch.arange(batch_size)[:, None]
    batch = torch.stack([torch.arange(512)] * batch_size) + offset
    output = patch(batch)

    assert torch.allclose(
        output[:, 1],
        torch.stack([torch.arange(patch_stride, patch_stride + patch_len)] * batch_size) + offset,
        atol=1e-5,
    )
    assert not torch.isnan(output).any()


def test_given_uneven_data_patch_operator_pads_and_output_is_correct():
    batch_size = 17
    patch_len = 16

    patch = Patch(patch_len, patch_len)

    batch = (torch.stack([torch.arange(512 - patch_len + 1)] * batch_size) + torch.arange(batch_size)[:, None]).float()
    output = patch(batch)

    assert output.shape == (batch_size, 512 // patch_len, patch_len)

    # check the first portion is padded
    assert torch.isnan(output[:, 0, :-1]).all()

    # check nowhere else is nan
    assert not torch.isnan(output[:, 1:]).any()


def test_when_instancenorm_applied_then_standardization_correct():
    inorm = InstanceNorm()

    input_ = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
        ]
    ).float()

    normalized, (loc, scale) = inorm(input_)

    assert normalized.shape == input_.shape
    assert torch.allclose(normalized[0], normalized[1])
    assert torch.allclose(loc.squeeze(), torch.tensor([3.0, 4.0]))
    assert torch.allclose(scale.squeeze(), torch.tensor(1.41421))


def test_when_instancenorm_applied_and_reversed_then_nans_preserved():
    inorm = InstanceNorm()

    input_ = torch.tensor(
        [
            [1, torch.nan, 3, 4, 5],
            [2, 3, 4, 5, torch.nan],
        ]
    ).float()

    normalized, (loc, scale) = inorm(input_)
    assert torch.allclose(normalized.isnan(), input_.isnan())

    output = inorm.inverse(normalized, (loc, scale))
    assert torch.allclose(output, input_, equal_nan=True)


def test_when_instancenorm_applied_and_reversed_then_output_correct():
    inorm = InstanceNorm()

    input_ = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 1000],
        ]
    ).float()

    normalized, loc_scale = inorm(input_)
    output = inorm.inverse(normalized, loc_scale)

    assert torch.allclose(output, input_)
