import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer
from transformers import logging as hf_logging

from ..constants import (
    AUTOMM,
    COLUMN,
    COLUMN_FEATURES,
    FEATURES,
    LABEL,
    LOGITS,
    MASKS,
    NER_ANNOTATION,
    TEXT_SEGMENT_IDS,
    TEXT_TOKEN_IDS,
    TEXT_VALID_LENGTH,
    TOKEN_WORD_MAPPING,
    WORD_OFFSETS,
)
from .huggingface_text import HFAutoModelForTextPrediction
from .utils import DummyLayer, assign_layer_ids, get_column_features, get_hf_config_and_model, init_weights

hf_logging.set_verbosity_error()

logger = logging.getLogger(AUTOMM)


class HFAutoModelForNER(HFAutoModelForTextPrediction):
    """
    Named entity recognition with huggingface backbones. Inherit from HFAutoModelForTextPrediction.
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str = "microsoft/deberta-v3-base",
        num_classes: Optional[int] = 0,
        pooling_mode: Optional[str] = "cls",
        gradient_checkpointing: Optional[bool] = False,
        pretrained: Optional[bool] = True,
    ):
        """
        Load a pretrained huggingface text transformer backbone.
        Parameters
        ----------
        prefix
            The model prefix.
        checkpoint_name
            Name of the checkpoint. We support loading checkpoint from
            Huggingface Models list: https://huggingface.co/models
            For example, you may use
                English backbones:
                    - 'bert-base-cased'
        num_classes
            The number of classes. 1 for a regression task.
        pooling_mode
            The pooling mode to be used, it is not used in the NER task.
        gradient_checkpointing
            Whether to enable gradient checkpointing
        pretrained
            Whether using the pretrained weights. If pretrained=True, download the pretrained model.
        """
        super().__init__(prefix, checkpoint_name, num_classes, pooling_mode, gradient_checkpointing, pretrained)
        logger.debug(f"initializing {checkpoint_name}")
        if self.config.model_type in {"gpt2", "roberta"}:
            # Refer to this PR: https://github.com/huggingface/transformers/pull/12116
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_name, add_prefix_space=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)

    @property
    def text_token_word_mapping_key(self):
        return f"{self.prefix}_{TOKEN_WORD_MAPPING}"

    @property
    def text_word_offsets_key(self):
        return f"{self.prefix}_{WORD_OFFSETS}"

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
        if self.disable_seg_ids:
            text_segment_ids = None
        else:
            text_segment_ids = batch[self.text_segment_ids_key]
        text_valid_length = batch[self.text_valid_length_key]
        steps = torch.arange(0, text_token_ids.shape[1]).type_as(text_valid_length)
        text_masks = (steps.reshape((1, -1)) < text_valid_length.reshape((-1, 1))).type_as(text_token_ids)

        if self.is_t5:
            # For the T5 model, we will only use the encoder to encode the sentence. This is adopted in
            # "Sentence-T5 (ST5): Scalable Sentence Encoders from Pre-trained Text-to-Text Models"
            # (https://aclanthology.org/2022.findings-acl.146.pdf).
            inputs_embeds = self.model.encoder.embed_tokens(text_token_ids)
            if self.gradient_checkpointing:
                # FIXME(?) This is a hack! We added a DummyLayer to ensure that the
                #  gradient checkpointing will assign output layer as require_grad=True
                #  Reference: https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
                inputs_embeds = self.dummy_layer(inputs_embeds)
            outputs = self.model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=text_masks,
            )
        else:
            outputs = self.model(
                input_ids=text_token_ids,
                token_type_ids=text_segment_ids,
                attention_mask=text_masks,
            )

        sequence_output = outputs.last_hidden_state
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)

        pooled_features = outputs.last_hidden_state[:, 0, :]

        logits = self.head(sequence_output)

        logits_label = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1)

        ret = {COLUMN_FEATURES: {FEATURES: {}, MASKS: {}}}
        column_features, column_feature_masks = get_column_features(
            batch=batch,
            column_name_prefix=self.text_column_prefix,
            features=outputs.last_hidden_state,
            valid_lengths=text_valid_length,
            cls_feature=pooled_features,
        )
        ret[COLUMN_FEATURES][FEATURES].update(column_features)
        ret[COLUMN_FEATURES][MASKS].update(column_feature_masks)

        ret.update(
            {
                LOGITS: logits,
                FEATURES: pooled_features,
                NER_ANNOTATION: logits_label,
                TOKEN_WORD_MAPPING: batch[self.text_token_word_mapping_key],
                WORD_OFFSETS: batch[self.text_word_offsets_key],
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
            "relative_attention_bias",
            "dummy_layer",
            "mask_emb",
            "word_embedding.weight",
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
