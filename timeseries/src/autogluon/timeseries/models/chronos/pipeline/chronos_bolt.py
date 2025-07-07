# Implements Chronos with T5 architecture but with patched inputs instead of
# per-time-step tokenization. a.k.a. Chronos-Bolt

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>, Lorenzo Stella <stellalo@amazon.com>, Caner Turkmen <atturkm@amazon.com>

import copy
import logging
import warnings
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.t5.modeling_t5 import (
    ACT2FN,
    T5Config,
    T5DenseActDense,
    T5DenseGatedActDense,
    T5LayerNorm,
    T5PreTrainedModel,
    T5Stack,
)
from transformers.utils import ModelOutput

from .base import BaseChronosPipeline, ForecastType

logger = logging.getLogger("autogluon.timeseries.models.chronos")


@dataclass
class ChronosBoltConfig:
    context_length: int
    prediction_length: int
    input_patch_size: int
    input_patch_stride: int
    quantiles: List[float]
    use_reg_token: bool = False


@dataclass
class ChronosBoltOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    quantile_preds: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    cross_attentions: Optional[torch.Tensor] = None


class Patch(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]

        if length % self.patch_size != 0:
            padding_size = (
                *x.shape[:-1],
                self.patch_size - (length % self.patch_size),
            )
            padding = torch.full(size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device)
            x = torch.concat((padding, x), dim=-1)

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        return x


class InstanceNorm(nn.Module):
    """
    See, also, RevIN. Apply standardization along the last dimension.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
            scale = torch.where(scale == 0, self.eps, scale)
        else:
            loc, scale = loc_scale

        return (x - loc) / scale, (loc, scale)

    def inverse(self, x: torch.Tensor, loc_scale: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        loc, scale = loc_scale
        return x * scale + loc


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = T5LayerNorm(out_dim)

    def forward(self, x: torch.Tensor):
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)

        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(out)
        return out


class SelectionBlock(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.layer_norm = T5LayerNorm(self.d_model)

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        self.dropout_layer = nn.Dropout(self.dropout)
        if config.is_gated_act:
            self.ff = T5DenseGatedActDense(config)
        else:
            self.ff = T5DenseActDense(config)

    def forward(self, hidden_states: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        normed_query = self.layer_norm(hidden_states)

        query_states = self.q(normed_query)
        key_states = self.k(memory)
        value_states = self.v(memory)

        query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        attn_weights = nn.functional.softmax(scores, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.inner_dim)
        attn_output = self.o(attn_output)

        hidden_states = hidden_states + self.dropout_layer(attn_output)
        hidden_states = self.ff(hidden_states)

        return hidden_states


class ChronosBoltModelForForecasting(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"input_patch_embedding\.",
        r"output_patch_embedding\.",
    ]
    _keys_to_ignore_on_load_unexpected = [r"lm_head.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(
        self,
        config: T5Config,
        dynamic_dims: int = 0,
        past_dynamic_dims: int = 0,
        static_dims: int = 0,
        static_cardinalities: Optional[List] = None,
        dynamic_cardinalities: Optional[List] = None,
        past_dynamic_cardinalities: Optional[List] = None,
    ):
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        super().__init__(config)
        self.model_dim = config.d_model

        # TODO: remove filtering eventually, added for backward compatibility
        config_fields = {f.name for f in fields(ChronosBoltConfig)}
        self.chronos_config = ChronosBoltConfig(
            **{k: v for k, v in config.chronos_config.items() if k in config_fields}
        )

        # Only decoder_start_id (and optionally REG token)
        if self.chronos_config.use_reg_token:
            config.reg_token_id = 1

        config.vocab_size = 2 if self.chronos_config.use_reg_token else 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # Input patch embedding layer
        self.input_patch_embedding = ResidualBlock(
            in_dim=self.chronos_config.input_patch_size * 2,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        # patching layer
        self.patch = Patch(
            patch_size=self.chronos_config.input_patch_size,
            patch_stride=self.chronos_config.input_patch_stride,
        )

        # instance normalization, also referred to as "scaling" in Chronos and GluonTS
        self.instance_norm = InstanceNorm()

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self._init_decoder(config)

        self.num_quantiles = len(self.chronos_config.quantiles)
        quantiles = torch.tensor(self.chronos_config.quantiles, dtype=self.dtype)
        self.register_buffer("quantiles", quantiles, persistent=False)

        self.output_patch_embedding = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=self.num_quantiles * self.chronos_config.prediction_length,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        # Covariates
        self.dynamic_dims = dynamic_dims
        self.past_dynamic_dims = past_dynamic_dims
        self.static_dims = static_dims
        self.static_cardinalities = static_cardinalities or []
        self.dynamic_cardinalities = dynamic_cardinalities or []
        self.past_dynamic_cardinalities = past_dynamic_cardinalities or []

        self.has_past_dynamic = (
            dynamic_dims + past_dynamic_dims + len(self.dynamic_cardinalities) + len(self.past_dynamic_cardinalities)
            > 0
        )
        self.has_future_dynamic = dynamic_dims + len(self.dynamic_cardinalities) > 0
        self.has_static = static_dims > 0 or len(self.static_cardinalities) > 0

        if self.has_past_dynamic or self.has_static:
            self.past_selector = SelectionBlock(config)
        if self.has_future_dynamic or self.has_static:
            self.future_selector = SelectionBlock(config)
            if self.has_future_dynamic:
                self.future_dynamic_patch_embedding = ResidualBlock(
                    in_dim=self.chronos_config.prediction_length * 2,
                    h_dim=config.d_ff,
                    out_dim=config.d_model,
                    act_fn_name=config.dense_act_fn,
                    dropout_p=config.dropout_rate,
                )

        if static_dims > 0:
            self.static_mapper = nn.Linear(static_dims, self.model_dim)
        if len(self.static_cardinalities) > 0:
            self.static_cat_embedders = nn.ModuleList(
                [nn.Embedding(c, self.model_dim) for c in self.static_cardinalities]
            )

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def _init_weights(self, module):
        super()._init_weights(module)
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, (self.__class__)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "static_cat_embedders"):
                for embedder in module.static_cat_embedders:
                    embedder.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "static_mapper"):
                module.static_mapper.weight.data.normal_(mean=0.0, std=factor * (self.static_dims**-0.5))
                if hasattr(module.static_mapper, "bias") and module.static_mapper.bias is not None:
                    module.static_mapper.bias.data.zero_()
        elif isinstance(module, ResidualBlock):
            module.hidden_layer.weight.data.normal_(
                mean=0.0,
                std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5),
            )
            if hasattr(module.hidden_layer, "bias") and module.hidden_layer.bias is not None:
                module.hidden_layer.bias.data.zero_()

            module.residual_layer.weight.data.normal_(
                mean=0.0,
                std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5),
            )
            if hasattr(module.residual_layer, "bias") and module.residual_layer.bias is not None:
                module.residual_layer.bias.data.zero_()

            module.output_layer.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.output_layer, "bias") and module.output_layer.bias is not None:
                module.output_layer.bias.data.zero_()
        elif isinstance(module, SelectionBlock):
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))

    def _past_feat_patch_and_embed(self, tensor: torch.Tensor) -> torch.Tensor:
        # patching
        mask = torch.isnan(tensor).logical_not().to(tensor.dtype)
        patched_input = self.patch(tensor)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_input[~(patched_mask > 0)] = 0.0
        # concat context and mask along patch dim
        patched_input = torch.cat([patched_input, patched_mask], dim=-1)
        return self.input_patch_embedding(patched_input)

    def _future_feat_patch_and_embed(self, tensor: torch.Tensor) -> torch.Tensor:
        mask = torch.isnan(tensor).logical_not().to(tensor.dtype)
        if self.chronos_config.prediction_length > tensor.shape[-1]:
            padding_shape = (*tensor.shape[:-1], self.chronos_config.prediction_length - tensor.shape[-1])
            tensor = torch.cat([tensor, torch.zeros(padding_shape).to(tensor)], dim=-1)
            mask = torch.cat([mask, torch.zeros(padding_shape).to(mask)], dim=-1)

        # patching
        patched_future = tensor.unsqueeze(-2)  # a single "patch"
        patched_mask = mask.unsqueeze(-2)  # a single "patch"
        patched_future[~(patched_mask > 0)] = 0.0
        # concat context and mask along patch dim
        patched_future = torch.cat([patched_future, patched_mask], dim=-1)
        return self.future_dynamic_patch_embedding(patched_future)

    def forward(
        self,
        context: torch.Tensor,  # [B, T]
        mask: Optional[torch.Tensor] = None,  # [B, T]
        target: Optional[torch.Tensor] = None,  # [B, H]
        target_mask: Optional[torch.Tensor] = None,  # [B, H]
        feat_static_real: Optional[torch.Tensor] = None,  # [B, D_sr]
        feat_static_cat: Optional[torch.Tensor] = None,  # [B, D_sc]
        feat_dynamic_real: Optional[torch.Tensor] = None,  # [B, T + H, D_dr]
        feat_dynamic_cat: Optional[torch.Tensor] = None,  # [B, T + H, D_dc]
        past_feat_dynamic_real: Optional[torch.Tensor] = None,  # [B, T, D_pr]
        past_feat_dynamic_cat: Optional[torch.Tensor] = None,  # [B, T, D_pc]
    ) -> ChronosBoltOutput:
        mask = mask.to(context.dtype) if mask is not None else torch.isnan(context).logical_not().to(context.dtype)

        batch_size, input_context_length = context.shape
        static_embeds = []
        past_dynamic_embeds = []
        future_dynamic_embeds = []
        # TODO: Add length asserts
        if self.dynamic_dims > 0:
            assert feat_dynamic_real is not None and feat_dynamic_real.shape[-1] == self.dynamic_dims
            feat_dynamic_real = feat_dynamic_real.transpose(1, 2)  # [B, D_dr, T + H]
            feat_dynamic_real_past_slice = feat_dynamic_real[..., :input_context_length]
            feat_dynamic_real_past_slice, dynamic_real_loc_scale = self.instance_norm(feat_dynamic_real_past_slice)
            feat_dynamic_real_future_slice = feat_dynamic_real[..., input_context_length:]

            feat_dynamic_real_future_slice, _ = self.instance_norm(
                feat_dynamic_real_future_slice, dynamic_real_loc_scale
            )
            past_dynamic_embeds.append(
                self._past_feat_patch_and_embed(feat_dynamic_real_past_slice)
            )  # [B, D_dr, P, d]
            future_dynamic_embeds.append(
                self._future_feat_patch_and_embed(feat_dynamic_real_future_slice)
            )  # [B, D_dr, P, d]
        if self.past_dynamic_dims > 0:
            assert past_feat_dynamic_real is not None and past_feat_dynamic_real.shape[-1] == self.past_dynamic_dims
            past_feat_dynamic_real = past_feat_dynamic_real.transpose(1, 2)  # [B, D_pr, T]
            past_feat_dynamic_real, _ = self.instance_norm(past_feat_dynamic_real)
            past_dynamic_embeds.append(self._past_feat_patch_and_embed(past_feat_dynamic_real))
        if self.static_dims > 0:
            assert feat_static_real is not None and feat_static_real.shape[-1] == self.static_dims
            # FIXME: Scaling?
            static_embeds.append(self.static_mapper(feat_static_real))
        if len(self.static_cardinalities) > 0:
            assert feat_static_cat is not None and feat_static_cat.shape[-1] == len(self.static_cardinalities)
            static_embeds.extend(
                [self.static_cat_embedders[j](feat_static_cat[:, j]) for j in range(len(self.static_cardinalities))]
            )
        if len(self.dynamic_cardinalities) > 0:
            assert feat_dynamic_cat is not None and feat_dynamic_cat.shape[-1] == len(self.dynamic_cardinalities)
            feat_dynamic_cat = feat_dynamic_cat.transpose(1, 2)  # [B, D_dc, T + H]
            feat_dynamic_cat_past_slice = feat_dynamic_cat[..., :input_context_length]
            feat_dynamic_cat_past_slice, dynamic_cat_loc_scale = self.instance_norm(
                feat_dynamic_cat_past_slice.float()
            )
            feat_dynamic_cat_future_slice = feat_dynamic_cat[..., input_context_length:]
            feat_dynamic_cat_future_slice, _ = self.instance_norm(feat_dynamic_cat_future_slice, dynamic_cat_loc_scale)
            past_dynamic_embeds.append(self._past_feat_patch_and_embed(feat_dynamic_cat_past_slice))
            future_dynamic_embeds.append(self._future_feat_patch_and_embed(feat_dynamic_cat_future_slice))
        if len(self.past_dynamic_cardinalities) > 0:
            assert past_feat_dynamic_cat is not None and past_feat_dynamic_cat.shape[-1] == len(
                self.past_dynamic_cardinalities
            )
            past_feat_dynamic_cat = past_feat_dynamic_cat.transpose(1, 2)  # [B, D_pc, T]
            past_feat_dynamic_cat, _ = self.instance_norm(past_feat_dynamic_cat.float())
            past_dynamic_embeds.append(self._past_feat_patch_and_embed(past_feat_dynamic_cat))

        # scaling
        context, loc_scale = self.instance_norm(context)

        # the scaling op above is done in 32-bit precision,
        # then the context is moved to model's dtype
        context = context.to(self.dtype)
        mask = mask.to(self.dtype)

        # patching
        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_context[~(patched_mask > 0)] = 0.0
        # concat context and mask along patch dim
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)

        # attention_mask = 1 if at least one item in the patch is observed
        attention_mask = patched_mask.sum(dim=-1) > 0  # (batch_size, patched_seq_length)

        input_embeds = self.input_patch_embedding(patched_context)  # [B, P, d]

        num_patches = input_embeds.shape[-2]
        if len(static_embeds) > 0:
            static_embeds = [embed[:, None, None, :] for embed in static_embeds]
            past_dynamic_embeds.extend([embed.expand(-1, 1, num_patches, -1) for embed in static_embeds])
            future_dynamic_embeds.extend(static_embeds)

        if self.has_past_dynamic or self.has_static:
            past_embeds = torch.cat([input_embeds.unsqueeze(1)] + past_dynamic_embeds, dim=1)  # [B, N_pf, P, d]
            num_past_feats = past_embeds.shape[1]
            past_embeds = past_embeds.transpose(1, 2).reshape(batch_size * num_patches, num_past_feats, self.model_dim)
            input_embeds = self.past_selector(
                input_embeds.view(batch_size * num_patches, 1, self.model_dim), past_embeds
            )
            input_embeds = input_embeds.view(batch_size, num_patches, self.model_dim)

        if self.chronos_config.use_reg_token:
            # Append [REG]
            reg_input_ids = torch.full(
                (batch_size, 1),
                self.config.reg_token_id,
                device=input_embeds.device,
            )
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat([attention_mask, torch.ones_like(reg_input_ids)], dim=-1)

        encoder_outputs = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
        )
        hidden_states = encoder_outputs[0]

        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.config.decoder_start_token_id,
            device=input_embeds.device,
        )
        decoder_input_embeds = self.shared(decoder_input_ids)  # [B, 1, d]

        if self.has_future_dynamic or self.has_static:
            future_embeds = torch.cat(
                [decoder_input_embeds.unsqueeze(1)] + future_dynamic_embeds, dim=1
            )  # [B, N_ff, 1, d]
            num_future_feats = future_embeds.shape[1]
            future_embeds = future_embeds.transpose(1, 2).reshape(batch_size, num_future_feats, self.model_dim)
            decoder_input_embeds = self.future_selector(
                decoder_input_embeds.view(batch_size, 1, self.model_dim), future_embeds
            )
            decoder_input_embeds = decoder_input_embeds.view(batch_size, 1, self.model_dim)

        sequence_output = self.decode(decoder_input_embeds, attention_mask, hidden_states)

        quantile_preds_shape = (
            batch_size,
            self.num_quantiles,
            self.chronos_config.prediction_length,
        )
        quantile_preds = self.output_patch_embedding(sequence_output).view(*quantile_preds_shape)

        loss = None
        if target is not None:
            # normalize target
            target, _ = self.instance_norm(target, loc_scale)
            target = target.unsqueeze(1)  # type: ignore
            assert self.chronos_config.prediction_length >= target.shape[-1]

            target = target.to(quantile_preds.device)
            target_mask = (
                target_mask.unsqueeze(1).to(quantile_preds.device) if target_mask is not None else ~torch.isnan(target)
            )
            target[~target_mask] = 0.0

            # pad target and target_mask if they are shorter than model's prediction_length
            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (*target.shape[:-1], self.chronos_config.prediction_length - target.shape[-1])
                target = torch.cat([target, torch.zeros(padding_shape).to(target)], dim=-1)
                target_mask = torch.cat([target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1)

            loss = (
                2
                * torch.abs(
                    (target - quantile_preds)
                    * ((target <= quantile_preds).float() - self.quantiles.view(1, self.num_quantiles, 1))
                )
                * target_mask.float()
            )
            loss = loss.mean(dim=-2)  # Mean over prediction horizon
            loss = loss.sum(dim=-1)  # Sum over quantile levels
            loss = loss.mean()  # Mean over batch

        # Unscale predictions
        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(batch_size, -1),
            loc_scale,
        ).view(*quantile_preds_shape)

        return ChronosBoltOutput(
            loss=loss,
            quantile_preds=quantile_preds,
        )

    def _init_decoder(self, config):
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

    def decode(
        self,
        decoder_input_embeds,
        attention_mask,
        hidden_states,
        output_attentions=False,
    ):
        """
        Parameters
        ----------
        decoder_input_embeds: torch.Tensor
            Decoder input embeddings. Shape (batch_size, 1, d_model)
        attention_mask: torch.Tensor
            Attention mask for the patched context. Shape (batch_size, patched_context_length), type: torch.int64
        hidden_states: torch.Tensor
            Hidden states returned by the encoder. Shape (batch_size, patched_context_length, d_model)

        Returns
        -------
        last_hidden_state
            Last hidden state returned by the decoder, of shape (batch_size, 1, d_model)
        """
        decoder_outputs = self.decoder(
            inputs_embeds=decoder_input_embeds,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )

        return decoder_outputs.last_hidden_state  # sequence_outputs, b x 1 x d_model


class ChronosBoltPipeline(BaseChronosPipeline):
    forecast_type: ForecastType = ForecastType.QUANTILES
    default_context_length: int = 2048
    # register this class name with this alias for backward compatibility
    _aliases = ["PatchedT5Pipeline"]

    def __init__(self, model: ChronosBoltModelForForecasting):
        super().__init__(inner_model=model)
        self.model = model

    @property
    def quantiles(self) -> List[float]:
        return self.model.config.chronos_config["quantiles"]

    def predict(  # type: ignore[override]
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        feat_static_real: Optional[torch.Tensor] = None,
        feat_static_cat: Optional[torch.Tensor] = None,
        feat_dynamic_real: Optional[torch.Tensor] = None,
        feat_dynamic_cat: Optional[torch.Tensor] = None,
        past_feat_dynamic_real: Optional[torch.Tensor] = None,
        past_feat_dynamic_cat: Optional[torch.Tensor] = None,
        prediction_length: Optional[int] = None,
        limit_prediction_length: bool = False,
    ):
        context_tensor = self._prepare_and_validate_context(context=context)

        model_context_length: int = self.model.config.chronos_config["context_length"]
        model_prediction_length: int = self.model.config.chronos_config["prediction_length"]
        if prediction_length is None:
            prediction_length = model_prediction_length

        if prediction_length > model_prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {model_prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg)

        predictions = []
        remaining = prediction_length

        # We truncate the context here because otherwise batches with very long
        # context could take up large amounts of GPU memory unnecessarily.
        if context_tensor.shape[-1] > model_context_length:
            context_tensor = context_tensor[..., -model_context_length:]
        if feat_static_real is not None:
            feat_static_real = feat_static_real.to(self.model.device)
        if feat_static_cat is not None:
            feat_static_cat = feat_static_cat.to(self.model.device)
        if feat_dynamic_real is not None:
            feat_dynamic_real = feat_dynamic_real[:, -model_context_length - prediction_length :].to(self.model.device)
        if feat_dynamic_cat is not None:
            feat_dynamic_cat = feat_dynamic_cat[:, -model_context_length - prediction_length :].to(self.model.device)
        if past_feat_dynamic_real is not None:
            past_feat_dynamic_real = past_feat_dynamic_real[:, -model_context_length:].to(self.model.device)
        if past_feat_dynamic_cat is not None:
            past_feat_dynamic_cat = past_feat_dynamic_cat[:, -model_context_length:].to(self.model.device)

        # TODO: We unroll the forecast of Chronos Bolt greedily with the full forecast
        # horizon that the model was trained with (i.e., 64). This results in variance collapsing
        # every 64 steps.
        while remaining > 0:
            with torch.no_grad():
                print(
                    f"{context_tensor.shape=}, {feat_dynamic_real.shape=}, {feat_dynamic_real[:, : context_tensor.shape[-1] + model_prediction_length].shape}"
                )
                prediction = self.model(
                    context=context_tensor.to(
                        device=self.model.device,
                        dtype=torch.float32,  # scaling should be done in 32-bit precision
                    ),
                    feat_static_real=feat_static_real,
                    feat_static_cat=feat_static_cat,
                    feat_dynamic_real=feat_dynamic_real[:, : context_tensor.shape[-1] + model_prediction_length]
                    if feat_dynamic_real is not None
                    else None,
                    feat_dynamic_cat=feat_dynamic_cat[:, : context_tensor.shape[-1] + model_prediction_length]
                    if feat_dynamic_cat is not None
                    else None,
                    past_feat_dynamic_real=past_feat_dynamic_real,
                    past_feat_dynamic_cat=past_feat_dynamic_cat,
                ).quantile_preds.to(context_tensor)

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            central_idx = torch.abs(torch.tensor(self.quantiles) - 0.5).argmin()
            central_prediction = prediction[:, central_idx]

            context_tensor = torch.cat([context_tensor, central_prediction], dim=-1)

        return torch.cat(predictions, dim=-1)[..., :prediction_length]

    def predict_quantiles(
        self, context: torch.Tensor, prediction_length: int, quantile_levels: List[float], **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # shape (batch_size, prediction_length, len(training_quantile_levels))
        predictions = (
            self.predict(context, prediction_length=prediction_length, **kwargs).detach().cpu().swapaxes(1, 2)
        )

        training_quantile_levels = self.quantiles

        if set(quantile_levels).issubset(set(training_quantile_levels)):
            # no need to perform intra/extrapolation
            quantiles = predictions[..., [training_quantile_levels.index(q) for q in quantile_levels]]
        else:
            # we rely on torch for interpolating quantiles if quantiles that
            # Chronos Bolt was trained on were not provided
            if min(quantile_levels) < min(training_quantile_levels) or max(quantile_levels) > max(
                training_quantile_levels
            ):
                logger.warning(
                    f"\tQuantiles to be predicted ({quantile_levels}) are not within the range of "
                    f"quantiles that Chronos-Bolt was trained on ({training_quantile_levels}). "
                    "Quantile predictions will be set to the minimum/maximum levels at which Chronos-Bolt "
                    "was trained on. This may significantly affect the quality of the predictions."
                )

            # TODO: this is a hack that assumes the model's quantiles during training (training_quantile_levels)
            # made up an equidistant grid along the quantile dimension. i.e., they were (0.1, 0.2, ..., 0.9).
            # While this holds for official Chronos-Bolt models, this may not be true in the future, and this
            # function may have to be revised.
            augmented_predictions = torch.cat(
                [predictions[..., [0]], predictions, predictions[..., [-1]]],
                dim=-1,
            )
            quantiles = torch.quantile(
                augmented_predictions, q=torch.tensor(quantile_levels, dtype=augmented_predictions.dtype), dim=-1
            ).permute(1, 2, 0)
        mean = predictions[:, :, training_quantile_levels.index(0.5)]
        return quantiles, mean

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Load the model, either from a local path or from the HuggingFace Hub.
        Supports the same arguments as ``AutoConfig`` and ``AutoModel``
        from ``transformers``.
        """
        # if optimization_strategy is provided, pop this as it won't be used
        kwargs.pop("optimization_strategy", None)

        config = AutoConfig.from_pretrained(*args, **kwargs)
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        context_length = kwargs.pop("context_length", None)
        if context_length is not None:
            config.chronos_config["context_length"] = context_length

        architecture = config.architectures[0]
        class_ = globals().get(architecture)

        # TODO: remove this once all models carry the correct architecture names in their configuration
        # and raise an error instead.
        if class_ is None:
            logger.warning(f"Unknown architecture: {architecture}, defaulting to ChronosBoltModelForForecasting")
            class_ = ChronosBoltModelForForecasting

        model = class_.from_pretrained(*args, **kwargs)
        return cls(model=model)
