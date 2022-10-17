import ast
import logging
import os
import warnings
from copy import deepcopy
from lib2to3.pgen2.token import OP
from typing import Any, Dict, List, Optional, Union

import numpy as np
from nptyping import NDArray
from omegaconf import DictConfig
from torch import nn
from transformers import AutoConfig, AutoTokenizer, BertTokenizer, CLIPTokenizer, ElectraTokenizer

from ..constants import (
    AUTOMM,
    CHOICES_IDS,
    LABEL,
    NER,
    NER_ANNOTATION,
    TEXT,
    TEXT_SEGMENT_IDS,
    TEXT_TOKEN_IDS,
    TEXT_VALID_LENGTH,
    TOKEN_WORD_MAPPING,
    WORD_OFFSETS,
)
from .collator import Pad, Stack
from .utils import process_ner_annotations, tokenize_ner_text

logger = logging.getLogger(AUTOMM)

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ALL_TOKENIZERS = {
    "bert": BertTokenizer,
    "clip": CLIPTokenizer,
    "electra": ElectraTokenizer,
    "hf_auto": AutoTokenizer,
}


class MultiModalProcessor:
    """
    Prepare multimodal data for the model specified by "prefix". For multiple models requiring multimodal data,
    we need to create a MultiModal for each related model so that they will have independent input.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer_name: Optional[str] = "hf_auto",
        max_len: Optional[int] = None,
        requires_column_info: bool = False,
    ):
        """
        Parameters
        ----------
        prefix
            The prefix connecting a processor to its corresponding model.
        tokenizer_name
            Name of the huggingface tokenizer type (default "hf_auto").
        max_len
            The maximum length of text tokens.
        requires_column_info
            Whether to require feature column information in dataloader.
        """
        self.prefix = model.prefix
        self.tokenizer_name = tokenizer_name
        self.requires_column_info = requires_column_info
        # Use the model's tokenizer if it exists.
        if hasattr(model, "tokenizer"):
            self.tokenizer = model.tokenizer
        else:
            self.tokenizer = self.get_pretrained_tokenizer(
                tokenizer_name=tokenizer_name,
                checkpoint_name=model.checkpoint_name,
            )

        if hasattr(self.tokenizer, "deprecation_warnings"):
            # Disable the warning "Token indices sequence length is longer than the specified maximum sequence..."
            # See https://github.com/huggingface/transformers/blob/6ac77534bfe97c00e0127bb4fc846ae0faf1c9c5/src/transformers/tokenization_utils_base.py#L3362
            self.tokenizer.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True

        if max_len is None or max_len <= 0:
            self.max_len = self.tokenizer.model_max_length
        else:
            if max_len < self.tokenizer.model_max_length:
                warnings.warn(
                    f"provided max length: {max_len} "
                    f"is smaller than {model.checkpoint_name}'s default: {self.tokenizer.model_max_length}"
                )
            self.max_len = min(max_len, self.tokenizer.model_max_length)
        logger.debug(f"text max length: {self.max_len}")

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
    def text_token_word_mapping_key(self):
        return f"{self.prefix}_{TOKEN_WORD_MAPPING}"

    @property
    def text_word_offsets_key(self):
        return f"{self.prefix}_{WORD_OFFSETS}"

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    def collate_fn(self, text_column_names: Optional[List] = None) -> Dict:
        """
        Collate multimodal features into a batch.
        This function will be used when creating Pytorch DataLoader.

        Returns
        -------
        A dictionary containing one model's collator function for multimodal data.
        """
        fn = {}
        if self.prefix == NER:
            fn.update(
                {
                    self.text_token_ids_key: Pad(pad_val=self.tokenizer.pad_token_id),
                    self.text_valid_length_key: Stack(),
                    self.text_segment_ids_key: Pad(pad_val=0),
                    self.text_token_word_mapping_key: Pad(pad_val=0),
                    self.text_word_offsets_key: Pad(pad_val=0),
                    self.label_key: Stack(),
                }
            )
        return fn

    @staticmethod
    def get_pretrained_tokenizer(
        tokenizer_name: str,
        checkpoint_name: str,
    ):
        """
        Load the tokenizer for a pre-trained huggingface checkpoint.

        Parameters
        ----------
        tokenizer_name
            The tokenizer type, e.g., "bert", "clip", "electra", and "hf_auto".
        checkpoint_name
            Name of a pre-trained checkpoint.

        Returns
        -------
        A tokenizer instance.
        """
        tokenizer_class = ALL_TOKENIZERS[tokenizer_name]
        return tokenizer_class.from_pretrained(checkpoint_name)

    def process_ner(
        self,
        sample: Dict[str, Union[int, float, list]],
        is_training: bool,
    ) -> Dict:
        """
        Process one NER sample's label and text features.
        New rules can be added if necessary.

        Parameters
        ----------
        labels
            One sample may have multiple labels.

        Returns
        -------
        A dictionary containing one NER sample's features and/or labels.
        """
        ret = {}
        ner_text = sample[TEXT]
        if is_training or NER_ANNOTATION in sample:
            ner_annotation = sample[NER_ANNOTATION]
            label, col_tokens, token_to_word_mappings, word_offsets = process_ner_annotations(
                ner_annotation, ner_text, self.tokenizer
            )
            ret.update({self.label_key: label})
        else:
            col_tokens, token_to_word_mappings, word_offsets = tokenize_ner_text(ner_text, self.tokenizer)
            ret.update({self.label_key: np.array([], dtype=np.int32)})

        ret.update(
            {
                self.text_token_ids_key: np.array(col_tokens.input_ids, dtype=np.int32),
                self.text_valid_length_key: sum(col_tokens.attention_mask),
                self.text_segment_ids_key: np.array(col_tokens.token_type_ids, dtype=np.int32),
                self.text_token_word_mapping_key: token_to_word_mappings,
                self.text_word_offsets_key: word_offsets,
            }
        )

        return ret

    def __call__(
        self,
        all: Dict[str, Union[NDArray[(Any,), Any], list]],
        is_training: bool,
    ) -> Dict:
        """
        Extract one sample's multimodal data.

        Parameters
        ----------
        all
            All the raw input data.
        is_training
            Whether to do processing in the training mode.

        Returns
        -------
        A dictionary containing one sample's features and/or labels.
        """
        return self.process_ner(all, is_training)
