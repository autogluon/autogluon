import collections
import logging
import os
import random
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from transformers import logging as hf_logging

from ..constants import (
    AUTOMM,
    CHOICES_IDS,
    COLUMN,
    COLUMN_FEATURES,
    FEATURES,
    LABEL,
    LM_TARGET,
    LOGITS,
    MASKS,
    TEMPLATE_LOGITS,
    TEXT_SEGMENT_IDS,
    TEXT_TOKEN_IDS,
    TEXT_VALID_LENGTH,
)
from .utils import DummyLayer, assign_layer_ids, get_column_features, get_pretrained_tokenizer

hf_logging.set_verbosity_error()

logger = logging.getLogger(__name__)


@lru_cache(None)
def warn_once(logger, msg: str):
    logger.warning(msg)


class TFewModel(nn.Module):
    """
    Implementation of T-Few (https://arxiv.org/pdf/2205.05638.pdf).
    Refer to https://github.com/r-three/t-few
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str = "bigscience/T0_3B",
        num_classes: Optional[int] = 0,
        length_norm: float = 1.0,  # Normalizes length to adjust for length bias in target template
        unlikely_loss: float = 1.0,  # Adds loss term that lowers probability of incorrect outputs
        mc_loss: float = 1.0,  # Adds multiple choice cross entropy loss
        gradient_checkpointing: Optional[bool] = False,
        low_cpu_mem_usage: Optional[bool] = False,
        pretrained: Optional[bool] = True,
        tokenizer_name: Optional[str] = "hf_auto",
    ):
        """
        Load a pretrained T5-based text transformer backbone.

        Parameters
        ----------
        prefix
            The model prefix.
        checkpoint_name
            Name of the checkpoint. We support loading T5ForConditionalGeneration checkpoints from
            Huggingface Models list: https://huggingface.co/models.
            We recommend using T0 backbones. For example, you may use
                - 'bigscience/T0_3B'
                - 'bigscience/T0p'
                - 'bigscience/T0pp'
        num_classes
            The number of classes. 1 for a regression task.
        length_norm
             Normalizes length to adjust for length bias in target template
        unlikely_loss
            Adds loss term that lowers probability of incorrect outputs
        mc_loss
            Adds multiple choice cross entropy loss
        gradient_checkpointing
            Whether to enable gradient checkpointing
        low_cpu_mem_usage
            Whether to turn on the optimization of reducing the peak CPU memory usage when loading the pretrained model.
        pretrained
            Whether using the pretrained weights. If pretrained=True, download the pretrained model.
        tokenizer_name
            Name of the huggingface tokenizer type.
        """
        super().__init__()
        logger.debug(f"initializing {checkpoint_name}")

        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes

        self.config = AutoConfig.from_pretrained(checkpoint_name)

        if pretrained:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_name, low_cpu_mem_usage=low_cpu_mem_usage)
        else:
            self.model = AutoModelForSeq2SeqLM.from_config(self.config)

        self.tokenizer_name = tokenizer_name
        self.tokenizer = get_pretrained_tokenizer(
            tokenizer_name=self.tokenizer_name,
            checkpoint_name=self.checkpoint_name,
        )
        self.eos_token = self.tokenizer.eos_token
        self.out_features = (
            self.model.config.hidden_size
        )  # required attribute for some features, e.g. data augmentation

        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.dummy_layer = DummyLayer()

        self.prefix = prefix

        self.mc_loss = mc_loss
        self.unlikely_loss = unlikely_loss
        self.length_norm = length_norm

        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    @property
    def text_token_ids_key(self):
        return f"{self.prefix}_{TEXT_TOKEN_IDS}"

    @property
    def text_segment_ids_key(self):
        return f"{self.prefix}_{TEXT_SEGMENT_IDS}"

    @property
    def text_valid_length_key(self):
        return f"{self.prefix}_{TEXT_VALID_LENGTH}"

    @property
    def input_keys(self):
        return [self.text_token_ids_key, self.text_valid_length_key, self.choices_key]

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    @property
    def choices_key(self):
        return f"{self.prefix}_{CHOICES_IDS}"

    @property
    def text_column_prefix(self):
        return f"{self.text_token_ids_key}_{COLUMN}"

    @property
    def text_feature_dim(self):
        return self.model.config.hidden_size

    def forward(
        self,
        text_token_ids: torch.Tensor,
        text_valid_length: torch.Tensor,
        choices_ids: Optional[torch.Tensor] = None,
        text_column_names: Optional[List[str]] = None,
        text_column_indices: Optional[List[torch.Tensor]] = None,
    ):
        """
        Parameters
        ----------
        text_token_ids : torch.Tensor
            Indices of input sequence tokens in the vocabulary.
        text_valid_length : torch.Tensor
            Valid length of the input text sequence.
        choices_ids : torch.Tensor, optional
            The choices ids for multiple-choices tasks.
        text_column_names : list of str, optional
            Names of the text columns.
        text_column_indices : list of torch.Tensor, optional
            Start and stop indices of the text columns.

        Returns
        -------
            A dictionary with logits and features.
        """
        # TODO: Bad style, check for choices in multimodal.data. Split TemplateEngine into TextTemplateEngine and LabelTemplateEngine.

        if not choices_ids.numel():
            warn_once(
                logger,
                msg="No target choices found in batch. Ensure that 'data.templates_turn_on=True' and that a valid preset or custom templates are provided.",
            )
            warn_once(logger, msg="Fallback to numerical representation of classes...")
            choices_ids = (
                self.tokenizer([str(i) for i in range(self.num_classes)], return_tensors="pt", padding=True)[
                    "input_ids"
                ]
                .repeat(text_token_ids.size(0), 1, 1)
                .to(text_token_ids)
            )

        assert (
            choices_ids.size(1) == self.num_classes
        ), f"Number of target choices is different from number of classes, but they must be the same. Please check template."

        bs = text_token_ids.size(0)
        # TODO(?) Currently does not support mixed-task batching, but can be added by adjusting the label_templates dict.

        bs, num_choices = choices_ids.size()[:2]
        flat_choices_ids = choices_ids.flatten(0, 1)

        text_masks = (text_token_ids != self.tokenizer.pad_token_id).float()

        inputs_embeds = self.model.encoder.embed_tokens(text_token_ids)

        if self.gradient_checkpointing:
            inputs_embeds = self.dummy_layer(inputs_embeds)

        # Forward input through the encoder
        encoder_hidden_states_or = self.model.encoder(inputs_embeds=inputs_embeds, attention_mask=text_masks)[0]
        encoder_hidden_states = encoder_hidden_states_or.unsqueeze(dim=1).repeat(1, num_choices, 1, 1).flatten(0, 1)

        attention_mask = text_masks.unsqueeze(dim=1).repeat(1, num_choices, 1).flatten(0, 1)
        decoder_input_ids = torch.cat([torch.zeros_like(flat_choices_ids[:, :1]), flat_choices_ids[:, :-1]], dim=1)
        decoder_attention_mask = (decoder_input_ids == decoder_input_ids).float()
        # Forward encoder output and target template as input for decoder
        model_output = self.model(
            attention_mask=attention_mask,
            encoder_outputs=[encoder_hidden_states],
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        model_output = model_output.logits

        target_template_logits = model_output  # Decoder Logits over the vocabulary for target template sequence

        lm_target = flat_choices_ids - 100 * (flat_choices_ids == self.tokenizer.pad_token_id).long()
        # Calculate entropy of target templates' logits to target template, i.e. how close the target template is to what
        # the model would predict, going from sentence start token (target_template_logits) to sentence end token (
        # lm_target)
        choices_scores = (
            F.cross_entropy(target_template_logits.flatten(0, 1), lm_target.flatten(0, 1), reduction="none")
            .view(bs, num_choices, -1)
            .sum(dim=-1)
        )
        # Add length normalization to adjust for target templates of different length
        if self.length_norm > 0:
            choices_scores = choices_scores / torch.pow(
                (choices_ids != self.tokenizer.pad_token_id).sum(dim=-1), self.length_norm
            )
        # Use the entropy score as the class "logit" scoring of T-Few.
        choices_scores = -choices_scores

        #  FIXME(?) Not sure having column features with the decoder vocabulary logits in T-Few makes sense
        batch = {
            self.text_token_ids_key: text_token_ids,
            self.text_valid_length_key: text_valid_length,
            self.choices_key: choices_ids,
        }
        if text_column_names:
            assert len(text_column_names) == len(text_column_indices), "invalid text column inputs"
            for idx, name in enumerate(text_column_names):
                batch[name] = text_column_indices[idx]
        column_features, column_feature_masks = get_column_features(
            batch=batch,
            column_name_prefix=self.text_column_prefix,
            features=model_output,
            valid_lengths=text_valid_length,
        )

        # needed to ensure compatibility to encoder-only pipelines
        features = encoder_hidden_states_or[:, 0, :]
        logits = choices_scores

        target_template_logits = target_template_logits.view(bs, num_choices, *target_template_logits.size()[1:])
        lm_target = lm_target.view(bs, num_choices, *lm_target.size()[1:])

        if column_features == {} or column_feature_masks == {}:
            return features, logits, target_template_logits, lm_target
        else:
            return features, logits, target_template_logits, lm_target, column_features, column_feature_masks

    def get_output_dict(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        target_template_logits: torch.Tensor,
        lm_target: torch.Tensor,
        column_features: Optional[Dict[str, torch.Tensor]] = None,
        column_feature_masks: Optional[Dict[str, torch.Tensor]] = None,
    ):
        ret = {COLUMN_FEATURES: {FEATURES: {}, MASKS: {}}}
        if column_features != None:
            ret[COLUMN_FEATURES][FEATURES].update(column_features)
            ret[COLUMN_FEATURES][MASKS].update(column_feature_masks)

        ret.update(
            {
                LOGITS: logits,  # needed for default crossentropy loss
                TEMPLATE_LOGITS: target_template_logits,  # needed for unlikelihood loss
                LM_TARGET: lm_target,  # needed for lm loss
                FEATURES: features,
            }
        )

        return {self.prefix: ret}

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
            "embeddings",
            "LayerNorm",
            "wte",
            "wpe",
            "shared.weight",
            "encoder.conv.conv",
            "dummy_layer",
        )
        post_encoder_patterns = ("head", "pooler", "ln_f", "final_layer_norm")
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
            name_to_id[n] = 1

        for name, id in name_to_id.items():  # no layer should be assigned zero id as zero id is finetuned
            if id == 0:
                name_to_id[name] = 1

        return name_to_id
