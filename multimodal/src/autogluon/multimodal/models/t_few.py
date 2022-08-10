import logging
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import logging as hf_logging
import random

from ..constants import (
    AUTOMM,
    COLUMN,
    COLUMN_FEATURES,
    FEATURES,
    LABEL,
    TEMPLATE_LOGITS,
    MASKS,
    TEXT_SEGMENT_IDS,
    TEXT_TOKEN_IDS,
    TEXT_VALID_LENGTH,
    LM_TARGET,
    LOGITS,
)
from .utils import assign_layer_ids, get_column_features

import json
hf_logging.set_verbosity_error()

logger = logging.getLogger(AUTOMM)


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x):
        return x + self.dummy_bias - self.dummy_bias


class TFewModel(nn.Module):
    """
    Implementation of T-Few (https://arxiv.org/pdf/2205.05638.pdf).
    Refer to https://github.com/r-three/t-few
    """

    def __init__(
        self,
        prefix: str,
        label_templates: dict = {},
        checkpoint_name: str = "bigscience/T0_3B",
        num_classes: Optional[int] = 0,
        length_norm: float = 1.0, # Normalizes length to adjust for length bias in target template
        unlikely_loss: float = 1.0, # Adds loss term that lowers probability of incorrect outputs
        mc_loss: float = 1.0, # Adds multiple choice cross entropy loss
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Load a pretrained T5-based text transformer backbone.

        Parameters
        ----------
        prefix
            The model prefix.
        label_templates
            Dictionary of textual templates to numerical class, as representations of these classes.
        checkpoint_name
            Name of the checkpoint. We support loading T5ForConditionalGeneration checkpoints from
            Huggingface Models list: https://huggingface.co/models.
            We recommend using T0 backbones. For example, you may use
                - 'bigscience/T0_3B'
                - 'bigscience/T0p'
                - 'bigscience/T0pp'
        num_classes
            The number of classes. 1 for a regression task.
        gradient_checkpointing
            Whether to enable gradient checkpointing
        label_templates
            A dictionary mapping the label ids to textual templates
        length_norm
             Normalizes length to adjust for length bias in target template
        unlikely_loss
            Adds loss term that lowers probability of incorrect outputs
        mc_loss
            Adds multiple choice cross entropy loss
        """
        super().__init__()
        logger.debug(f"initializing {checkpoint_name}")

        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes

        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_name)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
        self.eos_token = self.tokenizer.eos_token  # TODO: Do not hardcode

        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.dummy_layer = DummyLayer()
        self.prefix = prefix

        if len(label_templates) == 0:
            label_templates = {"{}".format(x): x for x in range(self.num_classes)}
            logger.info(f"Prompts for classes not specified. Fallback to default prompts (not recommended): {label_templates}")

        self.label_templates = label_templates
        self.label_templates_inverse = {y : [q for q, p in self.label_templates.items() if p == y] for x, y in self.label_templates.items()}

        self.mc_loss = mc_loss
        self.unlikely_loss = unlikely_loss
        self.length_norm = length_norm

        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

        if hasattr(self.model.config, "type_vocab_size") and self.model.config.type_vocab_size <= 1:
            # Disable segment ids for models like RoBERTa
            self.disable_seg_ids = True
        else:
            self.disable_seg_ids = False

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
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    @property
    def text_column_prefix(self):
        return f"{self.text_token_ids_key}_{COLUMN}"

    @property
    def text_feature_dim(self):
        return self.model.config.hidden_size

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

        bs = text_token_ids.size(0)
        # Sample uniformly target template descriptions from collection of descriptions
        # TODO: Currently does not support mixed-task batching, but can be added by adjusting the label_templates dict.
        selected_label_templates = [random.choice(y) for x, y in self.label_templates_inverse.items()][:self.num_classes]
        choices_ids = torch.tensor(self.tokenizer(selected_label_templates, padding=True)['input_ids']).type_as(text_token_ids)
        choices_ids = choices_ids.unsqueeze(0).repeat(text_token_ids.size(0), 1, 1)
        flat_choices_ids = choices_ids.flatten(0, 1)
        num_choices = len(selected_label_templates)

        # print(selected_label_templates)
        # print(batch[self.label_key])
        # print(self.tokenizer.batch_decode(text_token_ids))


        if self.disable_seg_ids:
            text_segment_ids = None
        else:
            text_segment_ids = batch[self.text_segment_ids_key]

        text_valid_length = batch[self.text_valid_length_key]
        text_masks = (text_token_ids != self.tokenizer.pad_token_id).float()

        inputs_embeds = self.model.encoder.embed_tokens(text_token_ids)

        if self.gradient_checkpointing:
            # FIXME(?) This is a hack! We added a DummyLayer to ensure that the
            #  gradient checkpointing will assign output layer as require_grad=True
            #  Reference: https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
            inputs_embeds = self.dummy_layer(inputs_embeds)

        # Foward input through the encoder
        encoder_hidden_states = self.model.encoder(inputs_embeds=inputs_embeds, attention_mask=text_masks)[0]
        encoder_hidden_states = encoder_hidden_states.unsqueeze(dim=1).repeat(1, num_choices , 1, 1).flatten(0, 1)

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

        target_template_logits = model_output.logits  # Decoder Logits over the vocabulary for target template sequence
        lm_target = flat_choices_ids - 100 * (flat_choices_ids == self.tokenizer.pad_token_id).long()
        # Calculate entropy of target templates' logits to target template, i.e. how close it target template to what
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

        ret = {COLUMN_FEATURES: {FEATURES: {}, MASKS: {}}}
        #  Not sure having column features with the decoder vocabulary logits in T-Few makes sense
        column_features, column_feature_masks = get_column_features(
            batch=batch,
            column_name_prefix=self.text_column_prefix,
            features=model_output.logits,
            valid_lengths=text_valid_length,
        )
        ret[COLUMN_FEATURES][FEATURES].update(column_features)
        ret[COLUMN_FEATURES][MASKS].update(column_feature_masks)

        ret.update(
            {
                LOGITS: choices_scores,  # needed for default crossentropy loss
                TEMPLATE_LOGITS: target_template_logits,  # needed for unlikelihood loss
                LM_TARGET: lm_target,  # needed for lm loss
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
            name_to_id[n] = 0

        return name_to_id
