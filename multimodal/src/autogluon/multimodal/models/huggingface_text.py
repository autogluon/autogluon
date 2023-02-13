import logging
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import AutoTokenizer
from transformers import logging as hf_logging
from transformers.models.t5 import T5PreTrainedModel

from ..constants import (
    AUTOMM,
    COLUMN,
    COLUMN_FEATURES,
    FEATURES,
    LABEL,
    LOGITS,
    MASKS,
    TEXT_SEGMENT_IDS,
    TEXT_TOKEN_IDS,
    TEXT_VALID_LENGTH,
)
from .utils import DummyLayer, assign_layer_ids, get_column_features, get_hf_config_and_model, init_weights

hf_logging.set_verbosity_error()

logger = logging.getLogger(__name__)


class HFAutoModelForTextPrediction(nn.Module):
    """
    Support huggingface text backbones.
    Refer to https://github.com/huggingface/transformers
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str = "microsoft/deberta-v3-base",
        num_classes: Optional[int] = 0,
        pooling_mode: Optional[str] = "cls",
        gradient_checkpointing: Optional[bool] = False,
        low_cpu_mem_usage: Optional[bool] = False,
        pretrained: Optional[bool] = True,
    ):
        """
        Load a pretrained huggingface text transformer backbone.

        Parameters
        ----------
        prefix
            The model prefix.
        checkpoint_name
            Name of the checkpoint or the local directory of a custom checkpoint.
            We support loading checkpoint from
            Huggingface Models list: https://huggingface.co/models
            For example, you may use
                English backbones:
                    - 'microsoft/deberta-v3-base'
                    - 'bert-base-uncased'
                    - 'google/electra-base-discriminator'
                    - 'distilroberta-base'
                Multilingual backbones:
                    - 'microsoft/mdeberta-v3-base'
                    - 'xlm-roberta-base'
        num_classes
            The number of classes. 1 for a regression task.
        pooling_mode
            The pooling mode for the Transformer. Can be "cls", or "mean"
        gradient_checkpointing
            Whether to enable gradient checkpointing
        low_cpu_mem_usage
            Whether to turn on the optimization of reducing the peak CPU memory usage when loading the pretrained model.
        pretrained
            Whether using the pretrained weights. If pretrained=True, download the pretrained model.
        """
        super().__init__()
        logger.debug(f"initializing {checkpoint_name}")
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes

        self.config, self.model = get_hf_config_and_model(
            checkpoint_name=checkpoint_name, pretrained=pretrained, low_cpu_mem_usage=low_cpu_mem_usage
        )
        self._hf_model_input_names = AutoTokenizer.from_pretrained(checkpoint_name).model_input_names

        if isinstance(self.model, T5PreTrainedModel):
            self.is_t5 = True
            # Remove the decoder in T5. We will only use the T5 encoder for extracting the embeddings
            del self.model.decoder
        else:
            self.is_t5 = False

        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            if self.is_t5:
                self.dummy_layer = DummyLayer()

        self.out_features = self.model.config.hidden_size

        self.head = nn.Linear(self.out_features, num_classes) if num_classes else nn.Identity()
        self.head.apply(init_weights)

        self.prefix = prefix
        self.pooling_mode = pooling_mode

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
    def input_keys(self):
        return [self.text_token_ids_key, self.text_segment_ids_key, self.text_valid_length_key]

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
        text_token_ids: torch.Tensor,
        text_segment_ids: torch.Tensor,
        text_valid_length: torch.Tensor,
        text_column_names: Optional[List[str]] = None,
        text_column_indices: Optional[List[torch.Tensor]] = None,
    ):
        """
        Parameters
        ----------
        text_token_ids : torch.Tensor
            Indices of input sequence tokens in the vocabulary.
        text_segment_ids : torch.Tensor
            Indices of input sequence segments.
        text_valid_length : torch.Tensor
            Valid length of the input text sequence.
        text_column_names : list of torch.Tensor, optional
            Names of the text columns.
        text_column_indices : list of torch.Tensor, optional
            Start and stop indices of the text columns.

        Returns
        -------
            A tuple that contains (pooled_features, logits, column_features, column_feature_masks)
        """
        if self.disable_seg_ids:
            text_segment_ids = None

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
            if "token_type_ids" in self._hf_model_input_names:
                outputs = self.model(
                    input_ids=text_token_ids,
                    token_type_ids=text_segment_ids,
                    attention_mask=text_masks,
                )
            else:
                outputs = self.model(
                    input_ids=text_token_ids,
                    attention_mask=text_masks,
                )
        if self.pooling_mode == "cls":
            pooled_features = outputs.last_hidden_state[:, 0, :]
        elif self.pooling_mode == "mean":
            pooled_features = (outputs.last_hidden_state * text_masks.unsqueeze(-1)).sum(1)
            sum_mask = text_masks.unsqueeze(-1).sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_features = pooled_features / sum_mask
        else:
            raise NotImplementedError(f"Pooling mode={self.pooling_mode} is not supported.")

        logits = self.head(pooled_features)
        last_hidden_state = outputs.last_hidden_state

        batch = {
            self.text_token_ids_key: text_token_ids,
            self.text_segment_ids_key: text_segment_ids,
            self.text_valid_length_key: text_valid_length,
        }
        if text_column_names:
            assert len(text_column_names) == len(text_column_indices), "invalid text column inputs"
            batch.update(**dict(zip(text_column_names, text_column_indices)))
        column_features, column_feature_masks = get_column_features(
            batch=batch,
            column_name_prefix=self.text_column_prefix,
            features=last_hidden_state,
            valid_lengths=text_valid_length,
            cls_feature=pooled_features,
        )

        if column_features == {} or column_feature_masks == {}:
            return pooled_features, logits
        else:
            return pooled_features, logits, column_features, column_feature_masks

    def get_output_dict(
        self,
        pooled_features: torch.Tensor,
        logits: torch.Tensor,
        column_features: Optional[List[torch.Tensor]] = None,
        column_feature_masks: Optional[List[torch.Tensor]] = None,
    ):
        ret = {COLUMN_FEATURES: {FEATURES: {}, MASKS: {}}}
        if column_features != None:
            ret[COLUMN_FEATURES][FEATURES].update(column_features)
            ret[COLUMN_FEATURES][MASKS].update(column_feature_masks)
        ret[LOGITS] = logits
        ret[FEATURES] = pooled_features
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

    def save(self, save_path: str = "./", tokenizers: Optional[dict] = None):
        self.model.save_pretrained(save_path)
        logger.info(f"Model weights for {self.prefix} are saved to {save_path}.")
        if self.prefix in tokenizers:
            tokenizers[self.prefix].save_pretrained(save_path)
            logger.info(f"Tokenizer {self.prefix} saved to {save_path}.")
