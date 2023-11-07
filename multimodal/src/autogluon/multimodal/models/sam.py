import logging
import os
import time
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from transformers import SamConfig, SamModel

from ..constants import COLUMN, IMAGE, IMAGE_VALID_NUM, LABEL, LOGITS, SEMANTIC_SEGMENTATION
from .utils import assign_layer_ids, freeze_model_layers

logger = logging.getLogger(__name__)


class SAMForSemanticSegmentation(nn.Module):
    """
    Support SAM for binary real-world semantic segmentation.
    Refer to https://huggingface.co/docs/transformers/main/model_doc/sam
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str,
        config: DictConfig,
        pretrained: Optional[bool] = True,
        frozen_layers: Optional[list] = None,
    ):
        """
        Load a pretrained Segment Anything Model (SAM).

        Parameters
        ----------
        prefix
            The prefix of the SAMForSemanticSegmentation model.
        checkpoint_name
            Name of the SAM checkpoint.
        pretrained
            Whether using the pretrained SAM models. If pretrained=True, download the pretrained model.
        frozen_layers
            A list of substrings of frozen layers' names.
        """

        super().__init__()
        self.prefix = prefix
        self.pretrained = pretrained
        self.checkpoint_name = checkpoint_name
        self.frozen_layers = frozen_layers

        self.device = None
        self.name_to_id = {}

        self._load_checkpoint(checkpoint_name)

        freeze_model_layers(self.model, self.frozen_layers)

        self.image_size = self.model.vision_encoder.image_size

        # for Binary Semantic Segmentation Tasks.
        self.model.mask_decoder.num_mask_tokens = 1
        mask_token_data = self.model.mask_decoder.mask_tokens.weight.data[0]
        self.model.mask_decoder.mask_tokens = nn.Embedding(1, self.model.mask_decoder.hidden_size)
        self.model.mask_decoder.mask_tokens.weight.data[0] = mask_token_data
        hyper_mlps = self.model.mask_decoder.output_hypernetworks_mlps[0]
        self.model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList([hyper_mlps])

        self.config = self.model.config

    def _load_checkpoint(self, checkpoint_name):
        if self.pretrained:
            self.model = SamModel.from_pretrained(checkpoint_name)
        else:
            configuration = SamConfig(name_or_path=checkpoint_name)
            self.model = SamModel(configuration)

    def save(self, save_path: str = "./"):
        self.model.save_pretrained(save_path)
        logger.info(f"Model weights for {self.prefix} is saved to {save_path}.")

    @property
    def image_key(self):
        return f"{self.prefix}_{IMAGE}"

    @property
    def image_valid_num_key(self):
        return f"{self.prefix}_{IMAGE_VALID_NUM}"

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    @property
    def image_column_prefix(self):
        return f"{self.image_key}_{COLUMN}"

    @property
    def image_feature_dim(self):
        return self.model.num_features

    def forward(
        self,
        batch,
    ):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with mask predictions.
        """
        rets = self.model(batch[self.image_key], multimask_output=False)
        pred_masks = rets.pred_masks[:, 0, :, :, :]
        pred_masks = F.interpolate(
            pred_masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        if self.training:
            return {self.prefix: {LOGITS: pred_masks}}
        else:
            return {self.prefix: {LOGITS: pred_masks, LABEL: batch[self.label_key]}}

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        In the AutoModel scenario, this function may not always return the correct result.
        Thus, you can use "print(json.dumps(name_to_id, indent=2))" to manually check whether
        the layer ids are reasonable.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = "model"
        pre_encoder_patterns = (
            "vision_encoder",
            "prompt_encoder",
        )
        post_encoder_patterns = ("mask_decoder",)

        names = [n for n, _ in self.named_parameters()]

        name_to_id, names = assign_layer_ids(
            names=names,
            pre_encoder_patterns=pre_encoder_patterns,
            post_encoder_patterns=post_encoder_patterns,
            model_pre=model_prefix,
        )
        if len(names) > 0:
            logger.debug(f"outer layers are treated as head: {names}")
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 0

        return name_to_id
