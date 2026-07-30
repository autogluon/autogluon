# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

from dataclasses import dataclass


@dataclass
class Toto2ModelConfig:
    """Architecture hyperparameters of a Toto 2.0 checkpoint, as stored in its ``config.json``."""

    patch_size: int
    d_model: int
    d_ff: int
    num_heads: int
    num_layers: int
    layer_group_size: int
    num_variate_layers_per_group: int
    variate_layer_first: bool
    residual_mult: float
    residual_attn_ratio: float
    qk_dim: int | None = None
    v_dim: int | None = None
    num_groups: int | None = None
    dropout_p: float = 0.0
    norm_eps: float = 5e-5
    attn_bias: bool = False
    mlp_bias: bool = False
    num_output_patches: int = 1
    pre_norm: bool = True
    qk_norm: bool = True
    norm_include_weight: bool = False
    qk_norm_include_weight: bool | None = None
    per_dim_scale: bool = False
    use_xpos: bool = False

    def __post_init__(self):
        if self.dropout_p != 0.0:
            # The inference-only implementation omits dropout entirely.
            raise ValueError(f"Only dropout_p=0.0 is supported, got {self.dropout_p}")
        if not self.pre_norm:
            raise ValueError("Only pre_norm=True is supported")
        if self.num_output_patches != 1:
            # All published checkpoints predict a single patch per token.
            raise ValueError(f"Only num_output_patches=1 is supported, got {self.num_output_patches}")

        if self.qk_norm_include_weight is None:
            self.qk_norm_include_weight = self.norm_include_weight
        self.num_groups = self.num_groups or self.num_heads
        self.qk_dim = self.qk_dim or self.d_model // self.num_heads
        self.v_dim = self.v_dim or self.qk_dim

        assert self.num_layers % self.layer_group_size == 0, (
            f"num_layers ({self.num_layers}) must be divisible by layer_group_size ({self.layer_group_size})"
        )
        assert self.num_heads > 0 and self.d_model % self.num_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        )
        assert self.num_heads >= self.num_groups and self.num_heads % self.num_groups == 0, (
            f"num_heads ({self.num_heads}) must be a multiple of num_groups ({self.num_groups})"
        )

    @property
    def heads_per_group(self) -> int:
        assert self.num_groups is not None
        return self.num_heads // self.num_groups

    @property
    def num_time_layers(self) -> int:
        """Number of time-attention layers, i.e. the number of layers that maintain a KV cache."""
        time_layers_per_group = self.layer_group_size - self.num_variate_layers_per_group
        return (self.num_layers // self.layer_group_size) * time_layers_per_group

    def is_variate_layer(self, layer_idx: int) -> bool:
        """Whether the layer at ``layer_idx`` attends across variates (rather than across time)."""
        position_in_group = layer_idx % self.layer_group_size
        if self.variate_layer_first:
            return position_in_group < self.num_variate_layers_per_group
        return position_in_group >= self.layer_group_size - self.num_variate_layers_per_group
