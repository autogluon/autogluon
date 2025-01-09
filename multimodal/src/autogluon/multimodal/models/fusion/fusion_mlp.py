import logging
from typing import List, Optional

import torch
from torch import nn

from ...constants import (
    AUG_LOGITS,
    FEATURES,
    LABEL,
    LOGITS,
    MULTIMODAL_FEATURES,
    MULTIMODAL_FEATURES_POST_AUG,
    MULTIMODAL_FEATURES_PRE_AUG,
    ORI_LOGITS,
    VAE_MEAN,
    VAE_VAR,
    WEIGHT,
)
from ..mlp import MLP
from ..utils import init_weights, run_model
from .base import AbstractMultimodalFusionModel

logger = logging.getLogger(__name__)


class MultimodalFusionMLP(AbstractMultimodalFusionModel):
    """
    Use MLP to fuse different models' features (single-modal and multimodal).
    Specifically, it adapts the features of each model to specified dimensions,
    concatenates the adapted features, and fuses the features through MLP.
    """

    def __init__(
        self,
        prefix: str,
        models: list,
        hidden_features: List[int],
        num_classes: int,
        adapt_in_features: Optional[str] = None,
        activation: Optional[str] = "gelu",
        dropout: Optional[float] = 0.5,
        normalization: Optional[str] = "layer_norm",
        aux_loss_weight: Optional[float] = None,
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
        activation
            Name of activation function.
        dropout
            Dropout probability.
        normalization
            Name of normalization function.
        aux_loss_weight
            The weight of individual models. For example, if we fuse the features of ViT, CLIP, and BERT,
            The loss will be computed as "loss = fusion_loss + aux_loss_weight(vit_loss + clip_loss + bert_loss)".
            Basically, it supports adding an auxiliary loss for each individual model.
        """
        super().__init__(
            prefix=prefix,
            models=models,
            aux_loss_weight=aux_loss_weight,
        )
        logger.debug(f"initializing {prefix} (MultimodalFusionMLP)")
        if aux_loss_weight is not None:
            assert aux_loss_weight >= 0
            logger.debug(f"auxiliary loss weight: {aux_loss_weight}")
        self.num_classes = num_classes
        self.augmenter = None

        raw_in_features = [per_model.out_features for per_model in models]
        if adapt_in_features is not None:
            if adapt_in_features == "min":
                base_in_feat = min(raw_in_features)
            elif adapt_in_features == "max":
                base_in_feat = max(raw_in_features)
            else:
                raise ValueError(f"unknown adapt_in_features: {adapt_in_features}")

            self.adapter = nn.ModuleList([nn.Linear(in_feat, base_in_feat) for in_feat in raw_in_features])

            in_features = base_in_feat * len(raw_in_features)
        else:
            self.adapter = nn.ModuleList([nn.Identity() for _ in range(len(raw_in_features))])
            in_features = sum(raw_in_features)

        assert len(self.adapter) == len(self.model)
        self.augmenter_in_features = in_features

        fusion_mlp = []
        for per_hidden_features in hidden_features:
            fusion_mlp.append(
                MLP(
                    in_features=in_features,
                    hidden_features=per_hidden_features,
                    out_features=per_hidden_features,
                    num_layers=1,
                    activation=activation,
                    dropout=dropout,
                    normalization=normalization,
                )
            )
            in_features = per_hidden_features
        self.fusion_mlp = nn.Sequential(*fusion_mlp)
        # in_features has become the latest hidden size
        self.head = nn.Linear(in_features, num_classes)
        # init weights
        self.adapter.apply(init_weights)
        self.fusion_mlp.apply(init_weights)
        self.head.apply(init_weights)

        self.out_features = in_features
        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    @property
    def input_keys(self):
        input_keys = []
        for m in self.model:
            assert hasattr(m, "input_keys"), f"invalid model {type(m)}, which doesn't have a 'input_keys' attribute"
            input_keys += m.input_keys
        return input_keys

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    def forward(
        self,
        *args,
    ):
        """

        Parameters
        ----------
        *args
            A list of torch.Tensor(s) containing the input mini-batch data. The fusion model doesn't need to
            directly access the mini-batch data since it aims to fuse the individual models'
            output features.

        Returns
        -------
        If "aux_loss_weight" is None, it returns dictionary containing the fusion model's logits and
        features. Otherwise, it returns a list of dictionaries collecting all the models' output,
        including the fusion model's.
        """
        multimodal_features = []
        multimodal_logits = []
        multimodal_features_pre_aug = None
        multimodal_features_post_aug = None
        vae_mean = None
        vae_var = None
        offset = 0
        for per_model, per_adapter in zip(self.model, self.adapter):
            per_model_args = args[offset : offset + len(per_model.input_keys)]
            batch = dict(zip(per_model.input_keys, per_model_args))
            per_output = run_model(per_model, batch)
            multimodal_features.append(
                per_adapter(per_output[per_model.prefix][FEATURES].to(per_adapter.weight.dtype))
            )
            multimodal_logits.append(per_output[per_model.prefix][LOGITS])
            offset += len(per_model.input_keys)

        # make sure the returned multimodal features contain unimodal encoder features
        multimodal_features_ret = multimodal_features
        multimodal_features = torch.cat(multimodal_features, dim=1)
        batch_size = multimodal_features.shape[0]
        if self.training and self.augmenter is not None:
            multimodal_features_pre_aug = multimodal_features.detach().clone()  # [bs, dim]
            multimodal_features_post_aug, vae_mean, vae_var = self.augmenter(multimodal_features_pre_aug)
            multimodal_features_post_aug_clone = multimodal_features_post_aug.clone()
            multimodal_features_post_aug_clone.register_hook(lambda grad: -grad * self.augmenter.adv_weight)
            multimodal_features = torch.cat([multimodal_features, multimodal_features_post_aug_clone], dim=0)

        features = self.fusion_mlp(multimodal_features)
        logits = self.head(features)
        ori_logits = logits[:batch_size].detach()  # detach the original logits when computing the consistency loss
        aug_logits = logits[batch_size:]

        return (
            features,
            logits,
            multimodal_logits,
            multimodal_features_ret,
            multimodal_features_pre_aug,
            multimodal_features_post_aug,
            ori_logits,
            aug_logits,
            vae_mean,
            vae_var,
        )

    def get_output_dict(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        multimodal_logits: List[torch.Tensor],
        multimodal_features: List[torch.Tensor],
        multimodal_features_pre_aug: torch.Tensor,
        multimodal_features_post_aug: torch.Tensor,
        ori_logits: torch.Tensor,
        aug_logits: torch.Tensor,
        vae_mean: torch.Tensor,
        vae_var: torch.Tensor,
    ):
        fusion_output = {
            self.prefix: {
                LOGITS: logits,
                FEATURES: features,
                MULTIMODAL_FEATURES: multimodal_features,
                MULTIMODAL_FEATURES_PRE_AUG: multimodal_features_pre_aug,
                MULTIMODAL_FEATURES_POST_AUG: multimodal_features_post_aug,
                ORI_LOGITS: ori_logits,
                AUG_LOGITS: aug_logits,
                VAE_MEAN: vae_mean,
                VAE_VAR: vae_var,
            }
        }
        # filter out None
        fusion_output = {self.prefix: {k: v for k, v in fusion_output[self.prefix].items() if v is not None}}
        if self.aux_loss_weight is not None:
            output = {}
            for per_model, per_logits in zip(self.model, multimodal_logits):
                per_output = {per_model.prefix: {}}
                per_output[per_model.prefix][WEIGHT] = torch.tensor(self.aux_loss_weight).to(per_logits.dtype)
                per_output[per_model.prefix][LOGITS] = per_logits
                output.update(per_output)
            fusion_output[self.prefix].update({WEIGHT: torch.tensor(1.0).to(logits)})
            output.update(fusion_output)
            return output
        else:
            return fusion_output
