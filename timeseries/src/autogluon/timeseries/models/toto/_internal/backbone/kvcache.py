# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

from dataclasses import dataclass, field

import torch

from .attention import TimeWiseMultiheadAttention

K = torch.Tensor
V = torch.Tensor
KV = tuple[torch.Tensor, torch.Tensor]


@dataclass
class KVCache:
    """
    Key/Value cache for storing intermediate attention values
    during multistep inference. Only stores KV cache for timewise layers, skipping spacewise layers.
    """

    batch_size: int
    num_variates: int
    transformer_layers: list
    num_layers: int
    embed_dim: int
    num_heads: int
    max_seq_len: int
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    use_memory_efficient_attention: bool = True

    _keys: torch.Tensor = field(init=False)
    _values: torch.Tensor = field(init=False)
    _current_idx: torch.Tensor = field(init=False)
    _layer_cache_map: torch.Tensor = field(init=False)

    def __post_init__(self):
        """
        - Determine timewise vs. spacewise layers and allocate KV only for timewise.
        - Create a fast tensor-based mapping from global layer_idx -> timewise layer_idx.
        """
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        head_dim = self.embed_dim // self.num_heads

        # Compute which layers are timewise
        time_layer_indices = [
            i
            for i in range(self.num_layers)
            if isinstance(self.transformer_layers[i].attention, TimeWiseMultiheadAttention)
        ]

        time_layer_count = max(1, len(time_layer_indices))  # handle edge case for no timewise layers
        # Allocate for only the timewise layers
        if self.use_memory_efficient_attention:
            shape = (
                time_layer_count,
                self.batch_size * self.num_variates,
                self.max_seq_len,
                self.num_heads,
                head_dim,
            )
        else:
            shape = (
                time_layer_count,
                self.batch_size * self.num_variates,
                self.num_heads,
                self.max_seq_len,
                head_dim,
            )
        self._keys = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self._values = torch.zeros_like(self._keys)
        self._current_idx = torch.zeros(time_layer_count, device=self.device, dtype=torch.int)
        # Build a tensor lookup for global -> timewise layer index (default to 0)
        self._layer_cache_map = torch.zeros((self.num_layers,), dtype=torch.int, device=self.device)
        for cache_idx, layer_idx in enumerate(time_layer_indices):
            self._layer_cache_map[layer_idx] = int(cache_idx)  # Assign correct indices

    def __getitem__(self, layer_idx: int) -> KV:
        cache_idx = int(self._layer_cache_map[layer_idx].item())
        end_idx = int(self._current_idx[cache_idx].item())

        if self.use_memory_efficient_attention:
            return self._keys[cache_idx, :, :end_idx, :, :], self._values[cache_idx, :, :end_idx, :, :]
        else:
            return self._keys[cache_idx, :, :, :end_idx, :], self._values[cache_idx, :, :, :end_idx, :]

    def current_len(self, cache_idx: int) -> int:
        return int(self._current_idx[cache_idx].item()) if self._current_idx.numel() > 0 else 0

    def seq_len(self, layer_idx: int) -> int:
        cache_idx = int(self._layer_cache_map[layer_idx].item())
        return self.current_len(cache_idx)

    def append(self, layer_idx: int, kv: KV):
        cache_idx = int(self._layer_cache_map[layer_idx].item())
        keys, values = kv

        # Validate dimensions
        assert keys.shape == values.shape, "keys and values must have the same shape"
        assert keys.shape[0] == self.batch_size * self.num_variates, (
            "keys and values must have batch_size * num_variates as their first dimension"
        )

        if self.use_memory_efficient_attention:
            assert keys.shape[2] == self.num_heads, "keys and values must have num_heads as their third dimension"
        else:
            assert keys.shape[1] == self.num_heads, "keys and values must have num_heads as their second dimension"
        assert keys.shape[3] == self.embed_dim // self.num_heads, (
            "keys and values must have head_dim as their fourth dimension"
        )

        start_idx = self._current_idx[cache_idx]
        if self.use_memory_efficient_attention:
            end_idx = start_idx + keys.shape[1]
        else:
            end_idx = start_idx + keys.shape[2]
        assert end_idx <= self.max_seq_len, (
            f"max_seq_len exceeded {end_idx} > {self.max_seq_len}, keys.shape: {keys.shape}"
        )

        if self.use_memory_efficient_attention:
            self._keys[cache_idx, :, start_idx:end_idx, :, :] = keys
            self._values[cache_idx, :, start_idx:end_idx, :, :] = values
        else:
            self._keys[cache_idx, :, :, start_idx:end_idx, :] = keys
            self._values[cache_idx, :, :, start_idx:end_idx, :] = values

        self._current_idx[cache_idx] = end_idx

    def reset(self):
        self._keys.zero_()
        self._values.zero_()
        self._current_idx.zero_()
