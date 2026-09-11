"""DenseLight architecture adapted from LightAutoML (Apache-2.0).

https://github.com/sb-ai-lab/LightAutoML

See ``DenseLightBlock`` / ``DenseLightModel`` in
``lightautoml/ml_algo/torch_based/nn_models.py``.

The defining trait is ``concat_input``: after the first hidden block, each
subsequent block receives the concatenation of its previous activations and
the original input features.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence, Union

import torch
import torch.nn as nn


class DenseLightBlock(nn.Module):
    """One DenseLight residual-style block: Linear → (BN) → Act → Dropout."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        drop_rate: float = 0.1,
        act_fun: type[nn.Module] = nn.LeakyReLU,
        use_bn: bool = True,
        bn_momentum: float = 0.1,
    ):
        super().__init__()
        self.features = nn.Sequential(OrderedDict())
        self.features.add_module("dense", nn.Linear(n_in, n_out, bias=(not use_bn)))
        if use_bn:
            self.features.add_module("norm", nn.BatchNorm1d(n_out, momentum=bn_momentum))
        self.features.add_module("act", act_fun())
        if drop_rate:
            self.features.add_module("dropout", nn.Dropout(p=drop_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class DenseLightNet(nn.Module):
    """DenseLight MLP with optional input concatenation into each hidden block."""

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        hidden_size: Union[int, Sequence[int]] = (512, 512),
        drop_rate: Union[float, Sequence[float]] = 0.1,
        act_fun: type[nn.Module] = nn.LeakyReLU,
        use_bn: bool = True,
        concat_input: bool = True,
        dropout_first: bool = True,
        bn_momentum: float = 0.1,
    ):
        super().__init__()

        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
        else:
            hidden_size = list(hidden_size)

        if isinstance(drop_rate, float):
            n_drops = len(hidden_size) + (1 if dropout_first else 0)
            drop_rate = [drop_rate] * n_drops
        else:
            drop_rate = list(drop_rate)

        expected_drops = len(hidden_size) + (1 if dropout_first else 0)
        if len(drop_rate) != expected_drops:
            raise ValueError(
                f"drop_rate length ({len(drop_rate)}) must equal "
                f"{expected_drops} (hidden layers + optional first dropout)."
            )

        self.concat_input = concat_input
        num_features = n_in

        self.features = nn.Sequential(OrderedDict())
        drops = list(drop_rate)
        if dropout_first and drops[0] > 0:
            self.features.add_module("dropout0", nn.Dropout(drops[0]))
            drops = drops[1:]

        for i, hid_size in enumerate(hidden_size):
            block = DenseLightBlock(
                n_in=num_features,
                n_out=hid_size,
                drop_rate=drops[i],
                act_fun=act_fun,
                use_bn=use_bn,
                bn_momentum=bn_momentum,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            if concat_input:
                # Next block input = previous hidden + original input (LAMA DenseLight).
                num_features = n_in + hid_size
            else:
                num_features = hid_size

        self.fc = nn.Linear(hidden_size[-1], n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        x0 = x.detach().clone()
        for name, layer in self.features.named_children():
            if self.concat_input and name not in ("dropout0", "denseblock1") and name.startswith("denseblock"):
                h = torch.cat([h, x0], dim=1)
            h = layer(h)
        return self.fc(h)
