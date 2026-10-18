# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
#
# This file also includes code derived from graphcore-research/unit-scaling
# (https://github.com/graphcore-research/unit-scaling), licensed under the Apache-2.0 License.
# Copyright 2023 Graphcore Ltd.

"""Minimal u-muP (unit scaling) primitives needed to run Toto 2.0 inference.

Toto 2.0 is trained with u-muP [Blake2024]_, where each op applies fixed multipliers to its output and to the
gradients flowing through it. The multipliers are part of the forward computation, so they must be reproduced exactly
to load the pretrained weights. This module ports the subset of ``dd_unit_scaling`` / ``unit_scaling`` that Toto 2.0
touches, so that neither package is required at runtime.

Omitted relative to the upstream packages: optimizers, distributed (world-size / gradient-accumulation aware) scale
factors, ``mup_type`` parameter tagging and ``DepthModuleList`` depth tagging (all of which only affect training), and
the general ``constraints`` machinery (only ``"to_output_scale"`` and ``None`` are used by Toto 2.0).

.. [Blake2024] Blake, Charlie, Eichenberg, Constantin et al.
    "u-muP: The Unit-Scaled Maximal Update Parametrization." (2024). https://arxiv.org/abs/2407.17465
"""

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F

# Output scale of the unit-scaled SiLU at ``mult=1`` under the "to_output_scale" constraint, i.e.
# ``logarithmic_interpolation(alpha=sigmoid(log(4)), lower=2, upper=sqrt(2 / (1 - 1 / pi)))``.
_SILU_SCALE = math.exp(
    (1 / (1 + 0.25)) * math.log((2 / (1 - 1 / math.pi)) ** 0.5) + (1 - 1 / (1 + 0.25)) * math.log(2.0)
)

# Empirically calibrated forward / backward scales of the unit-scaled softplus at ``mult=1``.
_SOFTPLUS_OUTPUT_SCALE = 1.0 / 0.52103
_SOFTPLUS_GRAD_INPUT_SCALE = 1.0 / 0.20833444


def _apply_constraint(constraint: Optional[str], output_scale: float, grad_input_scale: float) -> tuple[float, float]:
    """Tie the output and grad-input scales together according to ``constraint``.

    Only the constraints used by Toto 2.0 are supported: ``"to_output_scale"`` uses the output scale for both, and
    ``None`` leaves the two scales independent.
    """
    if constraint is None:
        return output_scale, grad_input_scale
    elif constraint == "to_output_scale":
        return output_scale, output_scale
    else:
        raise ValueError(f"Unsupported constraint {constraint!r}, expected 'to_output_scale' or None")


class _ScaledGrad(torch.autograd.Function):
    """Apply different scales in the forward and the backward pass.

    Uses the ``setup_context`` pattern so that ``torch.compile`` can trace through the op.
    """

    @staticmethod
    def forward(x: torch.Tensor, fwd_scale: Any, bwd_scale: Any) -> torch.Tensor:
        return fwd_scale * x

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
        ctx.bwd_scale = inputs[2]

    @staticmethod
    def backward(ctx: Any, grad_y: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        return ctx.bwd_scale * grad_y, None, None


def scale_fwd(input: torch.Tensor, scale: Any) -> torch.Tensor:
    """Scale a tensor in the forward pass only (the gradient is unchanged)."""
    return _ScaledGrad.apply(input, scale, 1.0)


def scale_bwd(input: torch.Tensor, scale: Any) -> torch.Tensor:
    """Scale a tensor's gradient in the backward pass only (the forward pass is the identity)."""
    return _ScaledGrad.apply(input, 1.0, scale)


def residual_split(input: torch.Tensor, tau: Any = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``input`` into a ``(residual, skip)`` pair with tau-weighted backward scaling."""
    denom = (1 + tau**2) ** 0.5
    return scale_bwd(input, tau / denom), scale_bwd(input, 1 / denom)


def residual_add(residual: torch.Tensor, skip: torch.Tensor, tau: Any = 1.0) -> torch.Tensor:
    """Combine a ``(residual, skip)`` pair with tau-weighted forward scaling."""
    denom = (1 + tau**2) ** 0.5
    return scale_fwd(residual, tau / denom) + scale_fwd(skip, 1 / denom)


def transformer_residual_scaling_rule(residual_mult: float, residual_attn_ratio: float):
    """Return ``fn(index, depth) -> tau``, the residual scaling rule from Appendix G of the u-muP paper.

    For a stack that alternates attention and MLP layers, this keeps the contribution of every attention layer (and
    separately, of every MLP layer) constant across depth, with ``residual_mult`` controlling the contribution of the
    residual layers relative to the input layer and ``residual_attn_ratio`` the ratio of attention to MLP.
    """
    alpha_mlp = residual_mult * (2 / (1 + residual_attn_ratio**2)) ** 0.5
    alpha_attn = residual_attn_ratio * alpha_mlp

    def tau(index: int, depth: int) -> float:
        num_attn, num_mlp = (index + 1) // 2, index // 2
        alpha = alpha_attn if index % 2 == 0 else alpha_mlp
        return alpha / (depth / 2 + num_attn * alpha_attn**2 + num_mlp * alpha_mlp**2) ** 0.5

    return tau


def silu(input: torch.Tensor) -> torch.Tensor:
    """Unit-scaled SiLU at ``mult=1`` under the "to_output_scale" constraint."""
    return scale_fwd(F.silu(scale_bwd(input, _SILU_SCALE)), _SILU_SCALE)


def softplus(input: torch.Tensor) -> torch.Tensor:
    """Unit-scaled softplus at ``mult=1``, scaled so that a standard normal input gives unit forward/backward scale."""
    return scale_fwd(F.softplus(scale_bwd(input, _SOFTPLUS_GRAD_INPUT_SCALE)), _SOFTPLUS_OUTPUT_SCALE)


def linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    constraint: Optional[str] = "to_output_scale",
    scale_power: tuple[float, float] = (0.5, 0.5),
) -> torch.Tensor:
    """Unit-scaled linear transformation.

    The output is scaled by ``1 / fan_in ** scale_power[0]`` and the input gradient by ``1 / fan_out ** scale_power[1]``
    (``fan_out``, per the u-muP reference implementation, counteracts the ``sqrt(fan_out)`` amplification of the
    backward pass). Gradient scaling of the weight and the bias is batch-size dependent and only matters for training,
    so it is omitted here.
    """
    fan_out, fan_in = weight.shape
    output_scale, grad_input_scale = _apply_constraint(
        constraint, 1.0 / fan_in ** scale_power[0], 1.0 / fan_out ** scale_power[1]
    )
    return scale_fwd(F.linear(scale_bwd(input, grad_input_scale), weight, bias), output_scale)


def rms_norm(
    input: torch.Tensor,
    normalized_shape: tuple[int, ...],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Unit-scaled RMS normalization over the trailing ``normalized_shape`` dimensions."""
    assert input.shape[-len(normalized_shape) :] == tuple(normalized_shape)
    dims = tuple(range(-len(normalized_shape), 0))
    # The mean of squares is accumulated in float32 for numerical stability, matching the reference implementation.
    rms = (input.float().pow(2).mean(dims, keepdim=True) + eps).sqrt().to(input.dtype)
    output = input / rms
    if weight is not None:
        output = output * scale_bwd(weight, (math.prod(normalized_shape) / input.numel()) ** 0.5)
    return output


def per_dim_scale(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Unit-scaled elementwise multiplication of ``input`` by ``weight``, broadcast along the last dimension.

    The forward scale ``0.52103`` undoes the forward scaling of :func:`softplus`, so that :class:`PerDimScale` is the
    identity at initialization (``weight`` parametrized by zeros).
    """
    grad_scale = (input.shape[-1] / input.numel()) ** 0.5
    return scale_fwd(scale_bwd(input, 0.52103) * scale_bwd(weight, grad_scale), 0.52103)


class Linear(torch.nn.Linear):
    """Unit-scaled ``torch.nn.Linear``. See :func:`linear`."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        constraint: Optional[str] = "to_output_scale",
        scale_power: tuple[float, float] = (0.5, 0.5),
    ):
        self.constraint = constraint
        self.scale_power = scale_power
        super().__init__(in_features, out_features, bias)

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return linear(input, self.weight, self.bias, self.constraint, self.scale_power)


class LinearReadout(Linear):
    """Unit-scaled linear layer for a final output projection: the output is scaled by ``1 / fan_in``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias, constraint=None, scale_power=(1.0, 0.5))


class RMSNorm(torch.nn.RMSNorm):
    """Unit-scaled ``torch.nn.RMSNorm``. See :func:`rms_norm`."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, include_weight: bool = False):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=include_weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return rms_norm(input, normalized_shape=self.normalized_shape, weight=self.weight, eps=self.eps)


class PerDimScale(torch.nn.Module):
    """Learned positive per-dimension scale factors, parametrized via softplus so that they stay positive.

    At initialization all factors equal 1.0, i.e. the module is the identity.
    """

    # Compensates for the softplus backward (4.8x) and the /log(2) (1.44x) amplification of the parameter gradient at
    # initialization, i.e. ``log(2) / (softplus_grad_input_scale * sigmoid(0))``.
    _param_grad_compensation = math.log(2.0) / (0.5 * _SOFTPLUS_GRAD_INPUT_SCALE)

    def __init__(self, dim: int):
        super().__init__()
        self.per_dim_scale = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = scale_bwd(self.per_dim_scale, self._param_grad_compensation)
        return per_dim_scale(input, (softplus(weight) / math.log(2.0)).to(input.dtype))
