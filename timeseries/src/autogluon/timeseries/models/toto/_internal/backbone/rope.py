# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

from typing import Optional

import torch
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from rotary_embedding_torch.rotary_embedding_torch import default


def exists(val):
    return val is not None


class TimeAwareRotaryEmbedding(RotaryEmbedding):
    """
    A variant of the rotary position embedding that (optionally) uses the time index
    to compute the sinusoidal and cosine embeddings. This is useful for
    time series data, where the time index is the most important positional
    information.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If the parent stored `freqs` as a Parameter, remove it and register as a buffer
        # Register buffer is needed for sharding with FSDP
        if hasattr(self, "freqs") and isinstance(self.freqs, torch.nn.Parameter):
            # Extract the underlying Tensor
            freqs_data = self.freqs.data

            # Remove `freqs` from the module's parameters
            self._parameters.pop("freqs")

            # Register as non-persistent buffer
            self.register_buffer("freqs", freqs_data, persistent=False)

    def rotate_queries_and_keys(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_dim: Optional[int] = None,
        seq_pos: Optional[torch.Tensor] = None,
        seq_pos_offset: int = 0,
    ):
        """
        This method is the same as the one on the base class, except it allows you to override
        the sequence position tensor with a custom one. It also removes the ability
        to cache the position encodings, since we have to compute them dynamically
        based on the timesteps in the input data.
        """
        if seq_dim is None:
            seq_dim = self.default_seq_dim

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = default(seq_pos, self.get_seq_pos(seq_len, dtype=dtype, device=device))
        seq = seq + seq_pos_offset  # type: ignore

        freqs = self.forward(seq)

        scale = self.get_scale(seq).to(dtype)

        # used for xformers
        if seq_dim == -3:
            num_heads = q.shape[-2]
            freqs = freqs.unsqueeze(1).expand(-1, num_heads, -1)
            scale = scale.unsqueeze(1).expand(-1, num_heads, -1)

        rotated_q = apply_rotary_emb(freqs, q, scale=scale, seq_dim=seq_dim)  # type: ignore
        rotated_k = apply_rotary_emb(freqs, k, scale=scale**-1, seq_dim=seq_dim)  # type: ignore

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def get_scale(self, t: torch.Tensor, seq_len: Optional[int] = None, offset=0):
        """
        This method is adapted closely from the base class, but it knows how to handle
        when `t` has more than 1 dim (as is the case when we're using time-aware RoPE, and have a different
        sequence position vector for each time series).
        """
        assert self.use_xpos

        power = (t - t.max(-1).values.unsqueeze(-1) // 2) / self.scale_base

        scale = self.scale ** rearrange(power, "... n -> ... n 1")  # type: ignore
        scale = torch.cat((scale, scale), dim=-1)

        return scale
