import logging
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from transformers import SamConfig, SamModel
from transformers.models.sam.modeling_sam import SamImageSegmentationOutput

from ..constants import CLASS_LABEL, CLASS_LOGITS, COLUMN, IMAGE, IMAGE_VALID_NUM, LABEL, LOGITS, MASK_LABEL
from .utils import assign_layer_ids, freeze_model_layers

logger = logging.getLogger(__name__)


def multi_class_mask_decoder_forward(
    self,
    image_embeddings: torch.Tensor,
    image_positional_embeddings: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
    multimask_output: bool,
    output_attentions: Optional[bool] = None,
    attention_similarity: torch.Tensor = None,
    target_embedding: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Modify the forward method of SamMaskDecoder for multi-class semantic segmentation
    based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/modeling_sam.py#L468.

    Args:
        image_embeddings (`torch.Tensor`):
            the embeddings from the image encoder
        image_positional_embedding (`torch.Tensor`):
            positional encoding with the shape of image_embeddings
        sparse_prompt_embeddings (`torch.Tensor`):
            The embeddings of the points and boxes
        dense_prompt_embeddings (`torch.Tensor`):
            the embeddings of the mask inputs
        multimask_output (bool):
            Whether to return multiple masks or a single mask.
        output_attentions (bool, *optional*):
            Whether or not to return the attentions tensors of all attention layers.
    """
    batch_size, num_channels, height, width = image_embeddings.shape
    point_batch_size = sparse_prompt_embeddings.shape[1]
    # Concatenate output tokens
    output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
    output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

    if sparse_prompt_embeddings.sum().item() != 0:
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
    else:
        tokens = output_tokens
    point_embeddings = tokens.to(self.iou_token.weight.dtype)

    # Expand per-image data in batch direction to be per-point
    image_embeddings = image_embeddings + dense_prompt_embeddings
    image_embeddings = image_embeddings.repeat(point_batch_size, 1, 1, 1)
    image_positional_embeddings = image_positional_embeddings.repeat(point_batch_size, 1, 1, 1)

    # Run the transformer, image_positional_embedding are consumed
    point_embedding, image_embeddings, attentions = self.transformer(
        point_embeddings=point_embeddings,
        image_embeddings=image_embeddings,
        image_positional_embeddings=image_positional_embeddings,
        attention_similarity=attention_similarity,
        target_embedding=target_embedding,
        output_attentions=output_attentions,
    )
    iou_token_out = point_embedding[:, :, 0, :]
    mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

    # Upscale mask embeddings and predict masks using the mask tokens
    image_embeddings = image_embeddings.transpose(2, 3).reshape(
        batch_size * point_batch_size, num_channels, height, width
    )

    upscaled_embedding = self.upscale_conv1(image_embeddings)
    upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
    upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

    ################ Modify the original code. We aim at returning a single mask for each object.
    ################ Original logic is to return multiple masks.
    ################ So we use an MLP network to process all the mask proposals instead of multiple networks.
    # hyper_in_list = []
    # for i in range(self.num_mask_tokens):
    #     current_mlp = self.output_hypernetworks_mlps[i]
    #     hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
    # hyper_in = torch.stack(hyper_in_list, dim=2)
    hyper_in = self.output_hypernetworks_mlps[0](mask_tokens_out)
    ################

    _, num_channels, height, width = upscaled_embedding.shape
    upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
    masks = (hyper_in @ upscaled_embedding).reshape(
        batch_size, point_batch_size, -1, height, width
    )  # bs, 1, num_tokens, h, w

    ################ New added class prediction logic.
    class_predictions = self.output_classifier_mlps(mask_tokens_out).reshape(
        batch_size, point_batch_size, -1, self.num_classes + 1
    )
    ################

    # Generate mask quality predictions
    iou_pred = self.iou_prediction_head(iou_token_out)

    outputs = (masks, iou_pred)

    if output_attentions:
        outputs = outputs + (attentions,)
    else:
        outputs = outputs + (None,)

    return outputs + (class_predictions,)


def multi_class_sam_model_forward(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    input_points: Optional[torch.FloatTensor] = None,
    input_labels: Optional[torch.LongTensor] = None,
    input_boxes: Optional[torch.FloatTensor] = None,
    input_masks: Optional[torch.LongTensor] = None,
    image_embeddings: Optional[torch.FloatTensor] = None,
    multimask_output: bool = True,
    attention_similarity: Optional[torch.FloatTensor] = None,
    target_embedding: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict=None,
    **kwargs,
) -> List[Dict[str, torch.Tensor]]:
    r"""
    Modify the forward method of SamModel for multi-class semantic segmentation
    based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/modeling_sam.py#L1279.

    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if pixel_values is None and image_embeddings is None:
        raise ValueError("Either pixel_values or image_embeddings must be provided.")

    if pixel_values is not None and image_embeddings is not None:
        raise ValueError("Only one of pixel_values and image_embeddings can be provided.")

    if input_points is not None and len(input_points.shape) != 4:
        raise ValueError(
            "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.",
            " got {}.".format(input_points.shape),
        )
    if input_boxes is not None and len(input_boxes.shape) != 3:
        raise ValueError(
            "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
            " got {}.".format(input_boxes.shape),
        )
    if input_points is not None and input_boxes is not None:
        point_batch_size = input_points.shape[1]
        box_batch_size = input_boxes.shape[1]
        if point_batch_size != box_batch_size:
            raise ValueError(
                "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                    point_batch_size, box_batch_size
                )
            )

    image_positional_embeddings = self.get_image_wide_positional_embeddings()
    # repeat with batch size
    batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
    image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

    vision_attentions = None
    vision_hidden_states = None

    if pixel_values is not None:
        vision_outputs = self.vision_encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeddings = vision_outputs[0]

        if output_hidden_states:
            vision_hidden_states = vision_outputs[1]
        if output_attentions:
            vision_attentions = vision_outputs[-1]

    if input_points is not None and input_labels is None:
        input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

    if input_points is not None and image_embeddings.shape[0] != input_points.shape[0]:
        raise ValueError(
            "The batch size of the image embeddings and the input points must be the same. ",
            "Got {} and {} respectively.".format(image_embeddings.shape[0], input_points.shape[0]),
            " if you want to pass multiple points for the same image, make sure that you passed ",
            " input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and ",
            " input_labels of shape (batch_size, point_batch_size, num_points_per_image)",
        )

    sparse_embeddings, dense_embeddings = self.prompt_encoder(
        input_points=input_points,
        input_labels=input_labels,
        input_boxes=input_boxes,
        input_masks=input_masks,
    )

    low_res_masks, iou_predictions, mask_decoder_attentions, class_predictions = self.mask_decoder(
        image_embeddings=image_embeddings,
        image_positional_embeddings=image_positional_embeddings,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
        attention_similarity=attention_similarity,
        target_embedding=target_embedding,
        output_attentions=output_attentions,
    )

    if not return_dict:
        output = (iou_predictions, low_res_masks)
        if output_hidden_states:
            output = output + (vision_hidden_states,)

        if output_attentions:
            output = output + (vision_attentions, mask_decoder_attentions)
        return output

    return (
        SamImageSegmentationOutput(
            iou_scores=iou_predictions,
            pred_masks=low_res_masks,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
            mask_decoder_attentions=mask_decoder_attentions,
        ),
        ################ New added. Return class predictions as well.
        class_predictions,
        ################
    )


class SAMForSemanticSegmentation(nn.Module):
    """
    Support SAM for semantic segmentation.
    Refer to https://huggingface.co/docs/transformers/main/model_doc/sam
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str,
        num_classes: int = 1,
        pretrained: Optional[bool] = True,
        frozen_layers: Optional[list] = None,
        num_mask_tokens: int = 1,
    ):
        """
        Load a pretrained Segment Anything Model (SAM).

        Parameters
        ----------
        prefix
            The prefix of the SAMForSemanticSegmentation model.
        checkpoint_name
            Name of the SAM checkpoint.
        num_classes
            The number of classes
        pretrained
            Whether using the pretrained SAM models. If pretrained=True, download the pretrained model.
        frozen_layers
            A list of substrings of frozen layers' names.
        num_mask_tokens
            The number of mask proposals.
        """

        super().__init__()
        self.prefix = prefix
        self.pretrained = pretrained
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes
        self.frozen_layers = frozen_layers

        self.device = None
        self.name_to_id = {}

        self._load_checkpoint(checkpoint_name)

        freeze_model_layers(self.model, self.frozen_layers)

        self.image_size = self.model.vision_encoder.image_size
        self.config = self.model.config

        self.model.mask_decoder.num_mask_tokens = num_mask_tokens
        mask_token_data = self.model.mask_decoder.mask_tokens.weight.data[0]
        self.model.mask_decoder.mask_tokens = nn.Embedding(num_mask_tokens, self.model.mask_decoder.hidden_size)
        for i in range(num_mask_tokens):
            self.model.mask_decoder.mask_tokens.weight.data[i] = mask_token_data
        hyper_mlps = self.model.mask_decoder.output_hypernetworks_mlps[0]
        self.model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList([hyper_mlps])
        if num_classes > 1:
            self.model.mask_decoder.num_classes = num_classes
            self.model.mask_decoder.output_classifier_mlps = nn.Linear(
                self.model.mask_decoder.hidden_size, num_classes + 1
            )

            mask_decoder_forward = multi_class_mask_decoder_forward.__get__(
                self.model.mask_decoder, self.model.mask_decoder.__class__
            )
            setattr(self.model.mask_decoder, "forward", mask_decoder_forward)

            sam_model_forward = multi_class_sam_model_forward.__get__(self.model, self.model.__class__)
            setattr(self.model, "forward", sam_model_forward)

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

    @property
    def mask_label_key(self):
        return f"{self.prefix}_{MASK_LABEL}"

    @property
    def class_label_key(self):
        return f"{self.prefix}_{CLASS_LABEL}"

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
        # binary
        if self.num_classes == 1:
            rets = self.model(batch[self.image_key], multimask_output=False)
            pred_masks = rets.pred_masks[:, 0, :, :, :]
            pred_masks = F.interpolate(
                pred_masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
            if self.training:
                return {self.prefix: {LOGITS: pred_masks}}
            else:
                return {self.prefix: {LOGITS: pred_masks, LABEL: batch[self.label_key]}}

        # multi-class
        else:
            rets, class_predictions = self.model(batch[self.image_key], multimask_output=False)
            pred_masks = rets.pred_masks[:, 0, :, :, :]
            pred_classes = class_predictions[:, 0, :, :]
            pred_masks = F.interpolate(
                pred_masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
            if self.training:
                return {self.prefix: {LOGITS: pred_masks, CLASS_LOGITS: pred_classes}}

            else:
                return {
                    self.prefix: {
                        LOGITS: pred_masks,
                        CLASS_LOGITS: pred_classes,
                        LABEL: batch[self.label_key],
                    }
                }

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
