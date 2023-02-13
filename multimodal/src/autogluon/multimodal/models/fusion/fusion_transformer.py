import logging
from typing import Optional

import torch
from torch import nn

from ...constants import AUTOMM, FEATURES, LABEL, LOGITS, WEIGHT
from ..ft_transformer import CLSToken, FT_Transformer
from ..utils import init_weights, run_model
from .base import AbstractMultimodalFusionModel

logger = logging.getLogger(__name__)


class MultimodalFusionTransformer(AbstractMultimodalFusionModel):
    """
    Use Transformer to fuse different models' features (single-modal and multimodal).
    Specifically, it adapts the features of each model to specified dimensions,
    concatenates the adapted features, and fuses the features through Transformer.
    """

    def __init__(
        self,
        prefix: str,
        models: list,
        hidden_features: int,
        num_classes: int,
        n_blocks: Optional[int] = 0,
        attention_n_heads: Optional[int] = 8,
        attention_initialization: Optional[str] = "kaiming",
        attention_normalization: Optional[str] = "layer_norm",
        attention_dropout: Optional[str] = 0.2,
        residual_dropout: Optional[str] = 0.0,
        ffn_activation: Optional[str] = "reglu",
        ffn_normalization: Optional[str] = "layer_norm",
        ffn_d_hidden: Optional[str] = 192,
        ffn_dropout: Optional[str] = 0.0,
        prenormalization: Optional[bool] = True,
        first_prenormalization: Optional[bool] = False,
        kv_compression_ratio: Optional[float] = None,
        kv_compression_sharing: Optional[str] = None,
        head_activation: Optional[str] = "relu",
        head_normalization: Optional[str] = "layer_norm",
        adapt_in_features: Optional[str] = None,
        loss_weight: Optional[float] = None,
        additive_attention: Optional[bool] = False,
        share_qv_weights: Optional[bool] = False,
    ):
        """
        Parameters
        ----------
        prefix
            The fusion model's prefix
        models
            The individual models whose output features will be fused.
        hidden_features
            A list of integers representing the hidden feature dimensions. For example,
            [512, 128, 64] indicates three hidden MLP layers with their corresponding output
            feature dimensions.
        num_classes
            The number of classes.
        n_blocks
            Number of the `FT_Transformer` blocks, which should be non-negative.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        kv_compression_ratio
            The compression ration to reduce the input sequence length.
        kv_compression_sharing
            If `true` the projections will share weights.
        head_activation
            Activation function type of the MLP layer.
        head_normalization
            Normalization scheme of the MLP layer.
        adapt_in_features
            Choice of how to adapt the features of each model. We now support
            - min
                Adapt all features to the minimum dimension. For example, if three models have
                feature dimensions [512, 768, 64], it will linearly map all the features to
                dimension 64.
            - max
                Adapt all features to the maximum dimension. For example, if three models have
                feature dimensions are [512, 768, 64], it will linearly map all the features to
                dimension 768.
        loss_weight
            The weight of individual models. For example, if we fuse the features of ViT, CLIP, and BERT,
            The loss will be computed as "loss = fusion_loss + loss_weight(vit_loss + clip_loss + bert_loss)".
            Basically, it supports adding an auxiliary loss for each individual model.
        additive_attention
            If 'true' the transformer will use additive attention with linear complexity to sequence length.
        share_qv_weights
            if 'true', then value and query transformation parameters are shared in additive attention.
        """
        super().__init__(
            prefix=prefix,
            models=models,
            loss_weight=loss_weight,
        )
        logger.debug("initializing MultimodalFusionTransformer")
        if loss_weight is not None:
            assert loss_weight > 0

        raw_in_features = [per_model.out_features for per_model in models]

        if adapt_in_features == "min":
            base_in_feat = min(raw_in_features)
        elif adapt_in_features == "max":
            base_in_feat = max(raw_in_features)
        else:
            raise ValueError(f"unknown adapt_in_features: {adapt_in_features}")

        self.adapter = nn.ModuleList([nn.Linear(in_feat, base_in_feat) for in_feat in raw_in_features])

        in_features = base_in_feat

        assert len(self.adapter) == len(self.model)

        self.fusion_transformer = FT_Transformer(
            d_token=in_features,
            n_blocks=n_blocks,
            attention_n_heads=attention_n_heads,
            attention_dropout=attention_dropout,
            attention_initialization=attention_initialization,
            attention_normalization=attention_normalization,
            ffn_d_hidden=ffn_d_hidden,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation,
            ffn_normalization=ffn_normalization,
            residual_dropout=residual_dropout,
            prenormalization=prenormalization,
            first_prenormalization=first_prenormalization,
            last_layer_query_idx=None,
            n_tokens=None,
            kv_compression_ratio=kv_compression_ratio,
            kv_compression_sharing=kv_compression_sharing,
            head_activation=head_activation,
            head_normalization=head_normalization,
            d_out=hidden_features,
            projection=False,
            additive_attention=additive_attention,
            share_qv_weights=share_qv_weights,
        )

        self.head = FT_Transformer.Head(
            d_in=in_features,
            d_out=num_classes,
            bias=True,
            activation=head_activation,
            normalization=head_normalization,
        )

        self.cls_token = CLSToken(
            d_token=in_features,
            initialization="uniform",
        )

        self.out_features = in_features

        # init weights
        self.adapter.apply(init_weights)
        self.head.apply(init_weights)
        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    def forward(
        self,
        batch: dict,
    ):
        multimodal_features = []
        output = {}
        for per_model, per_adapter in zip(self.model, self.adapter):
            per_output = run_model(per_model, batch)
            multimodal_feature = per_adapter(per_output[per_model.prefix][FEATURES])
            if multimodal_feature.ndim == 2:
                multimodal_feature = torch.unsqueeze(multimodal_feature, dim=1)
            multimodal_features.append(multimodal_feature)

            if self.loss_weight is not None:
                per_output[per_model.prefix].update(
                    {WEIGHT: torch.tensor(self.loss_weight).to(multimodal_features[0])}
                )
                output.update(per_output)

        multimodal_features = torch.cat(multimodal_features, dim=1)
        multimodal_features = self.cls_token(multimodal_features)
        features = self.fusion_transformer(multimodal_features)

        logits = self.head(features)
        fusion_output = {
            self.prefix: {
                LOGITS: logits,
                FEATURES: features,
            }
        }
        if self.loss_weight is not None:
            fusion_output[self.prefix].update({WEIGHT: torch.tensor(1.0).to(logits)})
            output.update(fusion_output)
            return output
        else:
            return fusion_output
