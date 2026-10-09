"""Deterministic backward for the SDPA operation used in Mitra fine-tuning."""

from threading import RLock

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

_determinism_lock = RLock()


class _DeterministicSDPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value):
        ctx.save_for_backward(query, key, value)
        return F.scaled_dot_product_attention(query, key, value)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # Unpack before enabling determinism: this may recompute a complete
        # checkpointed transformer layer, including its cuBLAS projections.
        inputs = [
            tensor.detach().requires_grad_(needed) for tensor, needed in zip(ctx.saved_tensors, ctx.needs_input_grad)
        ]
        with torch.enable_grad():
            output = F.scaled_dot_product_attention(*inputs)
        required = [tensor for tensor in inputs if tensor.requires_grad]
        # PyTorch's SDPA backward reads this flag when it runs. Scope it to
        # attention: enabling it around the whole fit also changes unrelated
        # kernels and requires process-wide cuBLAS environment configuration.
        with _determinism_lock:
            enabled = torch.are_deterministic_algorithms_enabled()
            warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
            try:
                torch.use_deterministic_algorithms(True)
                gradients = iter(torch.autograd.grad(output, required, grad_output))
            finally:
                torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        return tuple(next(gradients) if tensor.requires_grad else None for tensor in inputs)


def deterministic_scaled_dot_product_attention(query, key, value):
    if query.is_cuda and torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (query, key, value)):
        return _DeterministicSDPA.apply(query, key, value)
    return F.scaled_dot_product_attention(query, key, value)
