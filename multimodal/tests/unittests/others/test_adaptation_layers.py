import pytest
import torch
from torch import nn

from autogluon.multimodal.models.adaptation_layers import IA3Linear, IA3LoRALinear


def make_layer(layer_class, merge_weights=False, bias=True, **kwargs):
    if layer_class is IA3LoRALinear:
        kwargs["r"] = 2
    layer = layer_class(2, 3, merge_weights=merge_weights, bias=bias, **kwargs).double()
    with torch.no_grad():
        layer.weight.copy_(torch.arange(layer.weight.numel(), dtype=torch.float64).reshape_as(layer.weight) / 10)
        # Zero scaling must still allow the original weights to be recovered on returning to training.
        layer.lora_b.copy_(torch.tensor([[0.0], [2.0], [-1.0]]))
        if bias:
            layer.bias.copy_(torch.tensor([1.0, -2.0, 3.0]))
        if layer_class is IA3LoRALinear:
            layer.lora_A.fill_(0.1)
            layer.lora_B.fill_(0.2)
    return layer


@pytest.mark.parametrize("layer_class", [IA3Linear, IA3LoRALinear])
@pytest.mark.parametrize("merge_weights", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("evaluation", ["eval", "train_false", "parent_eval"])
def test_ia3_mode_changes_preserve_outputs_and_restore_weights(layer_class, merge_weights, bias, evaluation):
    layer = make_layer(layer_class, merge_weights=merge_weights, bias=bias)
    inputs = torch.arange(16, dtype=torch.float64).reshape(2, 4, 2) / 10
    expected = layer(inputs)
    original_weight = layer.weight.detach().clone()
    original_bias = layer.bias.detach().clone() if bias else None
    parent = nn.Sequential(layer)

    for _ in range(3):
        if evaluation == "eval":
            assert layer.eval() is layer
        elif evaluation == "train_false":
            assert layer.train(False) is layer
        else:
            assert parent.eval() is parent
        assert not layer.training
        assert layer.merged == merge_weights
        torch.testing.assert_close(layer(inputs), expected)
        assert layer.eval() is layer
        torch.testing.assert_close(layer(inputs), expected)

        assert layer.train() is layer
        assert layer.training and not layer.merged
        torch.testing.assert_close(layer.weight, original_weight, rtol=0, atol=0)
        if bias:
            torch.testing.assert_close(layer.bias, original_bias, rtol=0, atol=0)
        torch.testing.assert_close(layer(inputs), expected)


@pytest.mark.parametrize("layer_class", [IA3Linear, IA3LoRALinear])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("load_in_eval", [False, True])
def test_ia3_state_dict_roundtrip_from_merged_layer(layer_class, bias, load_in_eval):
    source = make_layer(layer_class, merge_weights=True, bias=bias)
    inputs = torch.ones(4, 2, dtype=torch.float64)
    expected = source(inputs)
    original_weight = source.weight.detach().clone()
    original_bias = source.bias.detach().clone() if bias else None
    source.eval()

    state = source.state_dict()
    torch.testing.assert_close(state["weight"], original_weight, rtol=0, atol=0)
    if bias:
        torch.testing.assert_close(state["bias"], original_bias, rtol=0, atol=0)
    assert not any("unmerged" in name for name in state)
    target = make_layer(layer_class, merge_weights=True, bias=bias)
    with torch.no_grad():
        target.weight.add_(1)
    if load_in_eval:
        target.eval()
    target.load_state_dict(state)

    assert target.training != load_in_eval
    torch.testing.assert_close(target(inputs), expected)
    target.eval()
    torch.testing.assert_close(target(inputs), expected)
    target.train()
    torch.testing.assert_close(target.weight, original_weight, rtol=0, atol=0)


@pytest.mark.parametrize("merge_weights", [False, True])
def test_ia3_lora_fan_in_fan_out_supports_non_square_weights(merge_weights):
    layer = make_layer(IA3LoRALinear, merge_weights=merge_weights, fan_in_fan_out=True)
    inputs = torch.ones(4, 2, dtype=torch.float64)
    assert layer.weight.shape == (2, 3)
    expected = layer(inputs)

    assert layer.eval() is layer
    torch.testing.assert_close(layer(inputs), expected)
    assert layer.train() is layer
    torch.testing.assert_close(layer(inputs), expected)


@pytest.mark.parametrize("layer_class", [IA3Linear, IA3LoRALinear])
def test_ia3_layers_disable_merging_by_default(layer_class):
    layer = layer_class(2, 3)
    assert layer.eval() is layer
    assert not layer.merged


def test_ia3_lora_mode_changes_update_dropout():
    layer = IA3LoRALinear(2, 3, lora_dropout=0.5)
    assert layer.eval() is layer
    assert not layer.lora_dropout.training
    assert layer.train() is layer
    assert layer.lora_dropout.training


@pytest.mark.parametrize("layer_class", [IA3Linear, IA3LoRALinear])
@pytest.mark.parametrize("merge_weights", [False, True])
def test_ia3_adapters_keep_gradients_after_evaluation(layer_class, merge_weights):
    layer = make_layer(layer_class, merge_weights=merge_weights)
    layer.eval().train()

    layer(torch.ones(4, 2, dtype=torch.float64)).square().sum().backward()

    assert layer.weight.grad is None
    assert layer.lora_b.grad is not None and torch.isfinite(layer.lora_b.grad).all()
    assert layer.bias.grad is not None and torch.isfinite(layer.bias.grad).all()
    if layer_class is IA3LoRALinear:
        assert layer.lora_A.grad is not None and torch.isfinite(layer.lora_A.grad).all()
        assert layer.lora_B.grad is not None and torch.isfinite(layer.lora_B.grad).all()


@pytest.mark.parametrize("layer_class", [IA3Linear, IA3LoRALinear])
def test_ia3_merged_backups_follow_dtype_changes(layer_class):
    layer = make_layer(layer_class, merge_weights=True)
    original_weight = layer.weight.detach().float().clone()
    original_bias = layer.bias.detach().float().clone()

    layer.eval().float().train()

    torch.testing.assert_close(layer.weight, original_weight, rtol=0, atol=0)
    torch.testing.assert_close(layer.bias, original_bias, rtol=0, atol=0)
