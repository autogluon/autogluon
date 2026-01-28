import json
import logging
import os
from typing import Optional, Union

import einops
import einx
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

# Try to import flash attention, but make it optional
try:
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

from torch.utils.checkpoint import checkpoint

from ..._internal.config.enums import Task
from ..._internal.models.base import BaseModel
from ..._internal.models.embedding import (
    Tab2DEmbeddingX,
    Tab2DEmbeddingYClasses,
    Tab2DEmbeddingYRegression,
    Tab2DQuantileEmbeddingX,
)

logger = logging.getLogger(__name__)


class Tab2D(BaseModel):
    def __init__(
        self,
        dim: int,
        dim_output: int,
        n_layers: int,
        n_heads: int,
        task: Union[str, Task],
        use_pretrained_weights: bool,
        path_to_weights: str,
        device: str = "cuda",  # Add device parameter
    ) -> None:
        super().__init__()

        self.dim = dim
        self.dim_output = dim_output
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.task = task
        self.device_type = device

        # Determine if we can use flash attention
        self.use_flash_attn = FLASH_ATTN_AVAILABLE and device.startswith("cuda")

        if isinstance(self.task, str):
            self.task = Task[self.task]

        self.x_quantile = Tab2DQuantileEmbeddingX(dim)
        self.x_embedding = Tab2DEmbeddingX(dim)

        if self.task == Task.CLASSIFICATION:
            self.y_embedding = Tab2DEmbeddingYClasses(dim, dim_output)  # type: nn.Module
        elif self.task == Task.REGRESSION:
            if self.dim_output == 1:
                self.y_embedding = Tab2DEmbeddingYRegression(dim)
            else:
                self.y_embedding = Tab2DEmbeddingYClasses(dim, dim_output)
        else:
            raise ValueError(f"Task {task} not supported")

        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(Layer(dim, n_heads, self.use_flash_attn))

        self.final_layer_norm = nn.LayerNorm(dim)

        self.final_layer = nn.Linear(dim, dim_output, bias=True)

        if use_pretrained_weights:
            if device == "cpu":
                # For CPU, use weights_only=False since CUDA checkpoints are incompatible with weights_only=True
                self.load_state_dict(torch.load(path_to_weights, weights_only=False, map_location=torch.device("cpu")))
            else:
                # For GPU, use weights_only=True for security
                self.load_state_dict(torch.load(path_to_weights, weights_only=True, map_location=device))
        else:
            self.init_weights()

    def forward(
        self,
        x_support: torch.Tensor,  # (b, n_s, f)
        y_support: torch.Tensor,  # (b, n_s)
        x_query: torch.Tensor,  # (b, n_q, f)
        padding_features: torch.Tensor,  # (b, f), "1" represents padding, "0" represents valid values
        padding_obs_support: torch.Tensor,  # (b, n_s)
        padding_obs_query__: torch.Tensor,  # (b, n_q)
    ):
        """
        x_support is (batch_size, n_observations_support, n_features)
        y_support is (batch_size, n_observations_support)

        x_query is (batch_size, n_observations_query, n_features)

        returns:

        y_query is (batch_size, n_observations_query, n_classes)

        syntax:
        b = batch size
        s = number of observations
        f = number of features
        d = dimension of embedding
        c = number of classes
        """

        x_query__ = x_query

        batch_size = x_support.shape[0]
        n_obs_support = x_support.shape[1]
        n_obs_query__ = x_query__.shape[1]

        x_support, x_query__ = self.x_quantile(x_support, x_query__, padding_obs_support, padding_features)
        x_support = self.x_embedding(x_support)  # (b, n_s, f, d)
        x_query__ = self.x_embedding(x_query__)  # (b, n_q, f, d)
        y_support, y_query__ = self.y_embedding(
            y_support, padding_obs_support, n_obs_query__
        )  # (b, n_s, 1, d), (b, n_q, 1, d)

        support, pack_support = einops.pack((y_support, x_support), "b s * d")  # (b, n_s, f+1, d)
        query__, pack_query__ = einops.pack((y_query__, x_query__), "b s * d")  # (b, n_q, f+1, d)

        padding_features_y = torch.zeros((batch_size, 1), device=padding_features.device, dtype=torch.bool)  # (b, 1)
        padding_features, _ = einops.pack((padding_features_y, padding_features), "b *")  # (b, f+1)

        if self.use_flash_attn:
            padder_support = Padder(support, padding_obs_support, padding_features)
            padder_query__ = Padder(query__, padding_obs_query__, padding_features)

            support = padder_support.base_to_obs(support)  # (n_valid_s, d)
            query__ = padder_query__.base_to_obs(query__)  # (n_valid_q, d)

            for layer in self.layers:
                support, query__ = checkpoint(
                    layer, support, query__, padder_support, padder_query__, use_reentrant=False
                )  # (n_valid_s, d), (n_valid_q, d)

            query__ = self.final_layer_norm(query__)
            query__ = self.final_layer(query__)  # (n_valid_q, d)

            query__ = padder_query__.obs_to_base(query__)  # (b, n_q, f+1, c)
        else:
            # For CPU/non-flash attention, work with standard tensor format
            for layer in self.layers:
                support, query__ = checkpoint(
                    layer,
                    support,
                    query__,
                    None,
                    None,
                    batch_size,
                    padding_obs_support,
                    padding_obs_query__,
                    padding_features,
                    use_reentrant=False,
                )

            query__ = self.final_layer_norm(query__)
            query__ = self.final_layer(query__)  # (b, n_q, f+1, c)

        y_query__, x_query__ = einops.unpack(query__, pack_query__, "b s * c")  # (b, n_q, 1, c), (b, n_q, f, c)

        if self.task == Task.REGRESSION:
            # output has shape (batch_size, n_observations_query, n_features, n_classes)
            # we want to remove the n_features dimension, and for regression, the n_classes dimension
            if self.dim_output == 1:
                y_query__ = y_query__[:, :, 0, 0]
            else:
                y_query__ = y_query__[:, :, 0, :]
        elif self.task == Task.CLASSIFICATION:
            y_query__ = y_query__[:, :, 0, :]
        else:
            raise ValueError(f"Task {self.task} not supported")

        return y_query__

    def init_weights(self) -> None:
        nn.init.normal_(self.x_embedding.x_embedding.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.x_embedding.x_embedding.bias, mean=0.0, std=1.0)
        nn.init.normal_(self.y_embedding.y_embedding.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.y_embedding.y_mask.weight, mean=0.0, std=1.0)

        # default PyTorch initialization for everything else

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)

        save_file(self.state_dict(), os.path.join(save_directory, "model.safetensors"))

        config = {
            "dim": self.dim,
            "dim_output": self.dim_output,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "task": str(self.task).upper(),
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, path_or_repo_id: str, device: str = "cuda") -> "Tab2D":
        config_path = hf_hub_download(repo_id=path_or_repo_id, filename="config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        model = cls(
            dim=config["dim"],
            dim_output=config["dim_output"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            task=config["task"],
            use_pretrained_weights=False,
            path_to_weights="",
            device=device,
        )

        weights_path = hf_hub_download(repo_id=path_or_repo_id, filename="model.safetensors")
        state_dict = load_file(weights_path, device=device)
        model.load_state_dict(state_dict)

        return model


class Padder(torch.nn.Module):
    def __init__(self, x: torch.Tensor, padding_mask: torch.Tensor, feature_mask: torch.Tensor) -> None:
        super().__init__()

        self.padding_mask = padding_mask
        self.feature_mask = feature_mask

        n_obs, n_feat = x.shape[1], x.shape[2]
        self.batch_size = x.shape[0]

        if not FLASH_ATTN_AVAILABLE:
            # CPU fallback: implement simplified padding logic without flash attention
            self._init_cpu_fallback(x, n_obs, n_feat)
            return

        # GPU path with flash attention
        self._init_flash_attn(x, n_obs, n_feat)

    def _init_cpu_fallback(self, x: torch.Tensor, n_obs: int, n_feat: int):
        """Initialize CPU-compatible version without flash attention dependencies."""
        # For CPU, we don't need the complex unpadding/padding logic
        # We'll implement pass-through methods that preserve tensor shapes
        self.cpu_mode = True

        # Store original shapes for reference
        self.original_shape = x.shape
        self.n_obs = n_obs
        self.n_feat = n_feat

        # These attributes won't be used in CPU mode but need to exist for compatibility
        self.cu_seqlens_o = None
        self.cu_seqlens_f = None
        self.cu_seqlens_fo = None
        self.cu_seqlens_of = None
        self.max_seqlen_in_batch_o = None
        self.max_seqlen_in_batch_f = None
        self.max_seqlen_in_batch_fo = None
        self.max_seqlen_in_batch_of = None

    def _init_flash_attn(self, x: torch.Tensor, n_obs: int, n_feat: int):
        """Initialize GPU version with flash attention."""
        self.cpu_mode = False

        # Original flash attention initialization logic
        x_o, self.indices_o, self.cu_seqlens_o, self.max_seqlen_in_batch_o, *_ = unpad_input(x, ~self.padding_mask)

        self.feature_mask_big = einops.repeat(self.feature_mask, "b f -> b s f", s=n_obs)
        self.feature_mask_big, _, _, _, *_ = unpad_input(self.feature_mask_big, ~self.padding_mask)
        x_of, self.indices_of, self.cu_seqlens_of, self.max_seqlen_in_batch_of, *_ = unpad_input(
            x_o, ~self.feature_mask_big
        )

        x_rearranged = einx.rearrange("b s f d -> b f s d", x)
        x_f, self.indices_f, self.cu_seqlens_f, self.max_seqlen_in_batch_f, *_ = unpad_input(
            x_rearranged, ~self.feature_mask
        )

        self.padding_mask_big = einops.repeat(self.padding_mask, "b s -> b f s", f=n_feat)
        self.padding_mask_big, _, _, _, *_ = unpad_input(self.padding_mask_big, ~self.feature_mask)
        x_fo, self.indices_fo, self.cu_seqlens_fo, self.max_seqlen_in_batch_fo, *_ = unpad_input(
            x_f, ~self.padding_mask_big
        )

        self.batch_size_f = x_f.shape[0]
        self.batch_size_o = x_o.shape[0]

        t = torch.arange(self.indices_fo.shape[0]).unsqueeze(1).to(x.device)
        self.obs_to_feat_indices = self.base_to_feat(self.obs_to_base(t)).squeeze(1)
        self.feat_to_obs_indices = self.base_to_obs(self.feat_to_base(t)).squeeze(1)

    def base_to_obs(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "cpu_mode") and self.cpu_mode:
            # CPU fallback: reshape for standard attention
            # Convert from (b, s, f, d) to (b*s, f*d) or similar flattened format
            b, s, f, d = x.shape
            return x.view(b * s, f * d)

        # GPU path with flash attention
        x = einx.rearrange("b s f d -> b f s d", x)
        x, _, _, _, *_ = unpad_input(x, ~self.feature_mask)
        x, _, _, _, *_ = unpad_input(x, ~self.padding_mask_big)
        return x

    def base_to_feat(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "cpu_mode") and self.cpu_mode:
            # CPU fallback: reshape for standard attention
            # Convert from (b, s, f, d) to (b*f, s*d) or similar flattened format
            b, s, f, d = x.shape
            return x.view(b * f, s * d)

        # GPU path with flash attention
        x, _, _, _, *_ = unpad_input(x, ~self.padding_mask)
        x, _, _, _, *_ = unpad_input(x, ~self.feature_mask_big)
        return x

    def obs_to_base(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "cpu_mode") and self.cpu_mode:
            # CPU fallback: reshape back to base format
            # This is the inverse of base_to_obs
            total_elements = x.numel()
            expected_d = self.original_shape[-1]  # last dimension
            b, s, f = self.batch_size, self.n_obs, self.n_feat
            return x.view(b, s, f, expected_d)

        # GPU path with flash attention
        x = pad_input(x, self.indices_fo, self.batch_size_f, self.max_seqlen_in_batch_fo)
        x = pad_input(x, self.indices_f, self.batch_size, self.max_seqlen_in_batch_f)
        x = einx.rearrange("b f s d -> b s f d", x)
        return x

    def feat_to_base(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "cpu_mode") and self.cpu_mode:
            # CPU fallback: reshape back to base format
            # This is the inverse of base_to_feat
            total_elements = x.numel()
            expected_d = self.original_shape[-1]  # last dimension
            b, s, f = self.batch_size, self.n_obs, self.n_feat
            return x.view(b, s, f, expected_d)

        # GPU path with flash attention
        x = pad_input(x, self.indices_of, self.batch_size_o, self.max_seqlen_in_batch_of)
        x = pad_input(x, self.indices_o, self.batch_size, self.max_seqlen_in_batch_o)
        return x

    def obs_to_feat(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "cpu_mode") and self.cpu_mode:
            # CPU fallback: simple pass-through or basic reshaping
            return x

        # GPU path with flash attention
        x = x[self.obs_to_feat_indices]
        return x

    def feat_to_obs(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "cpu_mode") and self.cpu_mode:
            # CPU fallback: simple pass-through or basic reshaping
            return x

        # GPU path with flash attention
        x = x[self.feat_to_obs_indices]
        return x


class Layer(torch.nn.Module):
    def __init__(self, dim: int, n_heads: int, use_flash_attn: bool) -> None:
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(dim)
        self.attention1 = MultiheadAttention(dim, n_heads, use_flash_attn)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 4, bias=True)
        self.linear2 = nn.Linear(dim * 4, dim, bias=True)

        self.layer_norm3 = nn.LayerNorm(dim)
        self.attention2 = MultiheadAttention(dim, n_heads, use_flash_attn)
        self.layer_norm4 = nn.LayerNorm(dim)
        self.linear3 = nn.Linear(dim, dim * 4, bias=True)
        self.linear4 = nn.Linear(dim * 4, dim, bias=True)

        self.use_flash_attn = use_flash_attn

    def forward(
        self,
        support: torch.Tensor,
        query__: torch.Tensor,
        padder_support: Optional[Padder],
        padder_query__: Optional[Padder],
        batch_size: Optional[int] = None,
        padding_obs_support: Optional[torch.Tensor] = None,
        padding_obs_query__: Optional[torch.Tensor] = None,
        padding_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
        support in 'obs' format
        query__ in 'obs' format

        Output:
        support in 'obs' format
        query__ in 'obs' format
        """

        if self.use_flash_attn and padder_support is not None and padder_query__ is not None:
            support_residual = support
            query___residual = query__

            support = self.layer_norm1(support)
            query__ = self.layer_norm1(query__)

            # attention across rows
            support_att = self.attention1(
                support,
                support,
                support,
                cu_seqlens_q=padder_support.cu_seqlens_fo,
                max_seqlen_q=padder_support.max_seqlen_in_batch_fo,
                cu_seqlens_k=padder_support.cu_seqlens_fo,
                max_seqlen_k=padder_support.max_seqlen_in_batch_fo,
            )
            query___att = self.attention1(
                query__,
                support,
                support,
                cu_seqlens_q=padder_query__.cu_seqlens_fo,
                max_seqlen_q=padder_query__.max_seqlen_in_batch_fo,
                cu_seqlens_k=padder_support.cu_seqlens_fo,
                max_seqlen_k=padder_support.max_seqlen_in_batch_fo,
            )

            support = support_residual + support_att
            query__ = query___residual + query___att

            support_residual = support
            query___residual = query__

            support = self.layer_norm2(support)
            query__ = self.layer_norm2(query__)

            support = self.linear1(support)
            query__ = self.linear1(query__)

            support = torch.nn.functional.gelu(support)
            query__ = torch.nn.functional.gelu(query__)

            support = self.linear2(support)
            query__ = self.linear2(query__)

            support = support_residual + support
            query__ = query___residual + query__

            support = padder_support.obs_to_feat(support)
            query__ = padder_query__.obs_to_feat(query__)

            support_residual = support
            query___residual = query__

            support = self.layer_norm3(support)
            query__ = self.layer_norm3(query__)

            # attention across features
            support = self.attention2(
                support,
                support,
                support,
                cu_seqlens_q=padder_support.cu_seqlens_of,
                max_seqlen_q=padder_support.max_seqlen_in_batch_of,
                cu_seqlens_k=padder_support.cu_seqlens_of,
                max_seqlen_k=padder_support.max_seqlen_in_batch_of,
            )
            query__ = self.attention2(
                query__,
                query__,
                query__,
                cu_seqlens_q=padder_query__.cu_seqlens_of,
                max_seqlen_q=padder_query__.max_seqlen_in_batch_of,
                cu_seqlens_k=padder_query__.cu_seqlens_of,
                max_seqlen_k=padder_query__.max_seqlen_in_batch_of,
            )

            support = support_residual + support
            query__ = query___residual + query__

            support_residual = support
            query___residual = query__

            support = self.layer_norm4(support)
            query__ = self.layer_norm4(query__)

            support = self.linear3(support)
            query__ = self.linear3(query__)

            support = torch.nn.functional.gelu(support)
            query__ = torch.nn.functional.gelu(query__)

            support = self.linear4(support)
            query__ = self.linear4(query__)

            support = support_residual + support
            query__ = query___residual + query__

            support = padder_support.feat_to_obs(support)
            query__ = padder_query__.feat_to_obs(query__)

            return support, query__
        else:
            # CPU/Standard attention path - ensure it matches the GPU logic exactly
            # Input format: (b, s, f+1, d) where f+1 includes the y embedding
            batch_size_actual, n_obs_support, n_feat_plus_one, dim = support.shape
            _, n_obs_query, _, _ = query__.shape

            if batch_size is None:
                batch_size = batch_size_actual

            # First attention block - attention across observations (rows)
            support_residual = support
            query___residual = query__

            support = self.layer_norm1(support)
            query__ = self.layer_norm1(query__)

            # Reshape for row attention: (b, s, f+1, d) -> (b*(f+1), s, d)
            support_flat = einops.rearrange(support, "b s f d -> (b f) s d")
            query___flat = einops.rearrange(query__, "b s f d -> (b f) s d")

            # attention across observations
            support_att_flat = self.attention1(support_flat, support_flat, support_flat)
            query___att_flat = self.attention1(query___flat, support_flat, support_flat)

            # Reshape back: (b*(f+1), s, d) -> (b, s, f+1, d)
            support_att = einops.rearrange(support_att_flat, "(b f) s d -> b s f d", b=batch_size)
            query___att = einops.rearrange(query___att_flat, "(b f) s d -> b s f d", b=batch_size)

            support = support_residual + support_att
            query__ = query___residual + query___att

            # First MLP block
            support_residual = support
            query___residual = query__

            support = self.layer_norm2(support)
            query__ = self.layer_norm2(query__)

            support = self.linear1(support)
            query__ = self.linear1(query__)

            support = torch.nn.functional.gelu(support)
            query__ = torch.nn.functional.gelu(query__)

            support = self.linear2(support)
            query__ = self.linear2(query__)

            support = support_residual + support
            query__ = query___residual + query__

            # Second attention block - attention across features
            support_residual = support
            query___residual = query__

            support = self.layer_norm3(support)
            query__ = self.layer_norm3(query__)

            # Reshape for feature attention: (b, s, f+1, d) -> (b*s, f+1, d)
            support_feat = einops.rearrange(support, "b s f d -> (b s) f d")
            query___feat = einops.rearrange(query__, "b s f d -> (b s) f d")

            # attention across features
            support_feat_att = self.attention2(support_feat, support_feat, support_feat)
            query___feat_att = self.attention2(query___feat, query___feat, query___feat)

            # Reshape back: (b*s, f+1, d) -> (b, s, f+1, d)
            support_feat_att = einops.rearrange(support_feat_att, "(b s) f d -> b s f d", b=batch_size)
            query___feat_att = einops.rearrange(query___feat_att, "(b s) f d -> b s f d", b=batch_size)

            support = support_residual + support_feat_att
            query__ = query___residual + query___feat_att

            # Second MLP block
            support_residual = support
            query___residual = query__

            support = self.layer_norm4(support)
            query__ = self.layer_norm4(query__)

            support = self.linear3(support)
            query__ = self.linear3(query__)

            support = torch.nn.functional.gelu(support)
            query__ = torch.nn.functional.gelu(query__)

            support = self.linear4(support)
            query__ = self.linear4(query__)

            support = support_residual + support
            query__ = query___residual + query__

            return support, query__


class MultiheadAttention(torch.nn.Module):
    def __init__(self, dim: int, n_heads: int, use_flash_attn: bool) -> None:
        super().__init__()

        self.use_flash_attn = use_flash_attn
        self.dim = dim
        self.n_heads = n_heads

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.o = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        b = batch size
        s = number of observations
        f = number of features
        t = flashattention-compressed sequences of (batch, observations, features)
        h = heads
        d = dimension of embedding

        input: (bsf, d) for flash attention or (b, s, d) for standard attention
        output: (bsf, d) for flash attention or (b, s, d) for standard attention
        """

        q = self.q(query)
        k = self.k(key)
        v = self.v(value)

        if self.use_flash_attn and cu_seqlens_q is not None:
            q = einops.rearrange(
                q, "t (h d) -> t h d", h=self.n_heads
            )  # (tokens, heads, dim), tokens is b*n*f w/o pad
            k = einops.rearrange(k, "t (h d) -> t h d", h=self.n_heads)
            v = einops.rearrange(v, "t (h d) -> t h d", h=self.n_heads)

            output = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,  # num_seq+1, either b*n (w/o pad)+1, or b*f (w/o pad)+1
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,  # max sequence length, either n or f
                max_seqlen_k=max_seqlen_k,
                deterministic=True,
            )

            output = einops.rearrange(output, "t h d -> t (h d)")
        else:
            # Standard scaled dot-product attention for CPU
            q = einops.rearrange(q, "b t (h d) -> b h t d", h=self.n_heads)
            k = einops.rearrange(k, "b t (h d) -> b h t d", h=self.n_heads)
            v = einops.rearrange(v, "b t (h d) -> b h t d", h=self.n_heads)

            output = F.scaled_dot_product_attention(q, k, v)
            output = einops.rearrange(output, "b h t d -> b t (h d)")

        output = self.o(output)

        return output
