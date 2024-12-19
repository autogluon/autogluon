import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...constants import FEATURES, LABEL, LOGITS, NER_ANNOTATION, NER_TEXT, TOKEN_WORD_MAPPING, WORD_OFFSETS
from ..mlp import MLP
from ..utils import run_model
from .base import AbstractMultimodalFusionModel

logger = logging.getLogger(__name__)


class MultimodalFusionNER(AbstractMultimodalFusionModel):
    """
    Use MLP to fuse different models' features (single-modal and multimodal) for NER.
    Specifically, it adapts the features of each model to specified dimensions,
    concatenates the adapted features, and fuses the features through MLP.
    """

    def __init__(
        self,
        prefix: str,
        models: list,
        hidden_features: List[int],
        num_classes: int,
        adapt_in_features: str = "max",
        activation: Optional[str] = "gelu",
        dropout_prob: Optional[float] = 0.5,
        normalization: Optional[str] = "layer_norm",
        loss_weight: Optional[float] = None,
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
        dropout_prob
            Dropout probability.
        normalization
            Name of normalization function.
        loss_weight
            The weight of individual models.
        """
        super().__init__(
            prefix=prefix,
            models=models,
            loss_weight=loss_weight,
        )
        logger.debug("initializing MultimodalFusionNER")

        if loss_weight is not None:
            assert loss_weight > 0
        self.num_classes = num_classes

        self.ner_model = None
        self.tokenizer = None
        other_models = []
        for per_model in models:
            if per_model.prefix != NER_TEXT:
                other_models.append(per_model)
            else:
                self.ner_model = per_model
                self.tokenizer = per_model.tokenizer
        self.other_models = nn.ModuleList(other_models)
        raw_in_features = [per_model.out_features for per_model in models if per_model.prefix != NER_TEXT]
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
        assert len(self.adapter) == len(self.other_models)
        fusion_mlp = []
        for per_hidden_features in hidden_features:
            fusion_mlp.append(
                MLP(
                    in_features=in_features,
                    hidden_features=per_hidden_features,
                    out_features=per_hidden_features,
                    num_layers=1,
                    activation=activation,
                    dropout_prob=dropout_prob,
                    normalization=normalization,
                )
            )
            in_features = per_hidden_features
        self.fusion_mlp = nn.Sequential(*fusion_mlp)
        self.head = nn.Linear(in_features + self.ner_model.out_features, num_classes)
        self.out_features = in_features
        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    @property
    def label_key(self):
        return f"{NER_TEXT}_{LABEL}"

    def forward(
        self,
        batch: dict,
    ):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data. The fusion model doesn't need to
            directly access the mini-batch data since it aims to fuse the individual models'
            output features.
        Returns
        -------
        It returns dictionary containing the fusion model's logits and features.
        """
        multimodal_features = []
        ner_output = run_model(self.ner_model, batch)
        for per_model, per_adapter in zip(self.other_models, self.adapter):
            per_output = run_model(per_model, batch)
            multimodal_features.append(per_adapter(per_output[per_model.prefix][FEATURES]))

        features = self.fusion_mlp(torch.cat(multimodal_features, dim=1))
        features = features.unsqueeze(dim=1).repeat(1, ner_output[self.ner_model.prefix][FEATURES].size()[1], 1)
        features = torch.cat((ner_output[self.ner_model.prefix][FEATURES], features), dim=-1)

        logits = self.head(features)
        logits_label = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1)
        fusion_output = {
            self.prefix: {
                LOGITS: logits,
                FEATURES: features,
                NER_ANNOTATION: logits_label,
                TOKEN_WORD_MAPPING: ner_output[self.ner_model.prefix][TOKEN_WORD_MAPPING],
                WORD_OFFSETS: ner_output[self.ner_model.prefix][WORD_OFFSETS],
            }
        }

        return fusion_output
