# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

from abc import ABC

import torch
import torch.nn.functional as F
from gluonts.torch.distributions import AffineTransformed
from gluonts.torch.distributions.studentT import StudentT


class DistributionOutput(ABC, torch.nn.Module):
    pass


class StudentTOutput(DistributionOutput):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.df = torch.nn.Linear(embed_dim, 1)
        self.loc_proj = torch.nn.Linear(embed_dim, 1)
        self.scale_proj = torch.nn.Linear(embed_dim, 1)

    def forward(self, inputs, loc=None, scale=None):
        eps = torch.finfo(inputs.dtype).eps
        df = 2.0 + F.softplus(self.df(inputs)).clamp_min(eps).squeeze(-1)
        base_loc = self.loc_proj(inputs).squeeze(-1)
        base_scale = F.softplus(self.scale_proj(inputs)).clamp_min(eps).squeeze(-1)

        base_dist = torch.distributions.StudentT(df, base_loc, base_scale, validate_args=False)  # type: ignore

        if loc is not None and scale is not None:
            return AffineTransformed(
                base_dist,
                loc=loc,
                scale=scale,
            )
        return base_dist


class MixtureOfStudentTsOutput(DistributionOutput):
    def __init__(
        self,
        embed_dim,
        k_components,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.k_components = k_components

        self.df = torch.nn.Linear(embed_dim, k_components)
        self.loc_proj = torch.nn.Linear(embed_dim, k_components)
        self.scale_proj = torch.nn.Linear(embed_dim, k_components)
        self.mixture_weights = torch.nn.Linear(embed_dim, k_components)

    def forward(self, inputs, loc=None, scale=None):
        df = 2.0 + F.softplus(self.df(inputs)).clamp_min(torch.finfo(inputs.dtype).eps)
        loc = self.loc_proj(inputs)
        scale = F.softplus(self.scale_proj(inputs)).clamp_min(torch.finfo(inputs.dtype).eps)
        logits = self.mixture_weights(inputs)
        probs = F.softmax(logits, dim=-1)
        components = StudentT(df, loc, scale)
        mixture_distribution = torch.distributions.Categorical(probs=probs)

        return torch.distributions.MixtureSameFamily(
            mixture_distribution,
            components,
        )
