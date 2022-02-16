import torch
import logging
from torch import nn
from transformers import CLIPModel
from .utils import assign_layer_ids
from ..constants import (
    IMAGE, TEXT_TOKEN_IDS, TEXT_VALID_LENGTH,
    LABEL, LOGITS, FEATURES, AUTOMM
)
from typing import Optional
from .utils import init_weights

logger = logging.getLogger(AUTOMM)


class CLIPForImageText(nn.Module):
    """
    Support the CLIP model.
    Refer to https://huggingface.co/docs/transformers/model_doc/clip
    """

    def __init__(
            self,
            prefix: str,
            checkpoint_name: str,
            num_classes: Optional[int] = 0,
    ):
        """
        Load the pretrained CLIP from huggingface transformers.

        Parameters
        ----------
        prefix
            The model prefix.
        checkpoint_name
            Name of the checkpoint.
        num_classes
            The number of classes. 1 for a regression task.
        """
        logger.debug(f"initializing {prefix}")
        super().__init__()
        self.model = CLIPModel.from_pretrained(checkpoint_name)
        self.out_features = self.model.config.projection_dim

        self.head = nn.Linear(self.out_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head.apply(init_weights)

        self.text_token_ids_key = f"{prefix}_{TEXT_TOKEN_IDS}"
        self.text_valid_length_key = f"{prefix}_{TEXT_VALID_LENGTH}"
        self.image_key = f"{prefix}_{IMAGE}"
        self.label_key = f"{prefix}_{LABEL}"

        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    def forward(
            self,
            batch: dict,
    ):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with logits and features.
        """
        text_token_ids = batch[self.text_token_ids_key]
        text_valid_length = batch[self.text_valid_length_key]
        image = batch[self.image_key]

        steps = torch.arange(0, text_token_ids.shape[1]).type_as(text_valid_length)
        text_masks = (steps.reshape((1, -1)) < text_valid_length.reshape((-1, 1))).type_as(text_token_ids)

        # Image batch has shape (batch_size, image_num, 3, height, width).
        # Currently, we only support image_num=1 for CLIP input.
        if image.dim() == 5 and image.shape[1] == 1:
            image = torch.squeeze(image, dim=1)
        assert image.dim() == 4

        assert torch.equal(text_valid_length, text_masks.sum(dim=-1))

        image_features = self.model.get_image_features(pixel_values=image)
        text_features = self.model.get_text_features(input_ids=text_token_ids,
                                                     attention_mask=text_masks)
        # Here we add up the text and image embeddings
        features = image_features + text_features
        logits = self.head(features)

        return {
            LOGITS: logits,
            FEATURES: features,
        }

    def get_layer_ids(self,):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefixes = ["model.text_model", "model.vision_model", "model"]
        # later model prefixes can't starts with the early ones
        for i, model_pre in enumerate(model_prefixes):
            for model_pre2 in model_prefixes[i+1:]:
                if model_pre2.startswith(model_pre):
                    raise ValueError(
                        f"{model_pre} is a substring of {model_pre2}. "
                        f"Need to swap them in {model_prefixes}."
                    )

        pre_encoder_patterns = ("embeddings", "pre")
        post_encoder_patterns = ("head", "final", "post", "logit", "project")
        names = [n for n, _ in self.named_parameters()]

        name_to_id = {}
        for per_prefix in model_prefixes:
            per_model_name_to_id, names = assign_layer_ids(
                names=names,
                pre_encoder_patterns=pre_encoder_patterns,
                post_encoder_patterns=post_encoder_patterns,
                model_pre=per_prefix,
            )
            name_to_id.update(per_model_name_to_id)

        if len(names) > 0:
            logger.debug(f"outer layers are treated as head: {names}")
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 0

        return name_to_id
