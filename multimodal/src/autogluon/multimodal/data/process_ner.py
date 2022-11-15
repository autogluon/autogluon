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
    NER_TEXT,
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


class NerProcessor:
    """
    Prepare NER data for the model specified by "prefix".
    """

    def __init__(
        self,
        model: nn.Module,
        max_len: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        prefix
            The prefix connecting a processor to its corresponding model.
        """
        self.model = model
        self.prefix = model.prefix
        self.tokenizer = None
        self.tokenizer_name = model.prefix
        self.max_len = max_len

        if self.prefix == NER_TEXT:
            self.tokenizer = model.tokenizer

            if max_len is None or max_len <= 0:
                self.max_len = self.tokenizer.model_max_length
            else:
                if max_len < self.tokenizer.model_max_length:
                    warnings.warn(
                        f"provided max length: {max_len} "
                        f"is smaller than {model.checkpoint_name}'s default: {self.tokenizer.model_max_length}"
                    )
                self.max_len = min(max_len, self.tokenizer.model_max_length)

            self.tokenizer.model_max_length = self.max_len

    @staticmethod
    def get_pretrained_tokenizer(
        tokenizer_name: str,
        checkpoint_name: str,
    ):
        return AutoTokenizer.from_pretrained(checkpoint_name)

    def collate_fn(self, text_column_names: Optional[List] = None) -> Dict:
        """
        Collate multimodal features into a batch.
        This function will be used when creating Pytorch DataLoader.

        Returns
        -------
        A dictionary containing one model's collator function for multimodal data.
        """
        fn = {}
        if self.prefix == NER_TEXT:
            fn.update(
                {
                    self.model.text_token_ids_key: Pad(pad_val=self.tokenizer.pad_token_id),
                    self.model.text_valid_length_key: Stack(),
                    self.model.text_segment_ids_key: Pad(pad_val=0),
                    self.model.text_token_word_mapping_key: Pad(pad_val=0),
                    self.model.text_word_offsets_key: Pad(pad_val=0),
                    self.model.label_key: Stack(),
                }
            )
        return fn

    def process_ner(
        self,
        all_features: Dict[str, Union[int, float, list]],
        feature_modalities: Dict[str, Union[int, float, list]],
        is_training: bool,
    ) -> Dict:
        """
        Process one NER sample's label and text features.
        New rules can be added if necessary.

        Parameters
        ----------
        all_features
            All features including text and ner annotations.
        feature_modalities
            The modality of the feature columns.

        Returns
        -------
        A dictionary containing one NER sample's features and/or labels.
        """
        ret = {}
        # overwrite model_max_length for standalone checkpoints if it's not specified.
        if self.max_len is not None and self.tokenizer.model_max_length > 10**6:
            self.tokenizer.model_max_length = self.max_len
        text_column, annotation_column = None, None
        for column_name, column_modality in feature_modalities.items():
            if column_modality == TEXT:
                text_column = column_name
            if column_modality == NER_ANNOTATION:
                annotation_column = column_name

        ner_text = all_features[text_column]
        if is_training or annotation_column is not None:
            ner_annotation = all_features[annotation_column]
            label, col_tokens, token_to_word_mappings, word_offsets = process_ner_annotations(
                ner_annotation, ner_text, self.tokenizer
            )
            ret.update({self.model.label_key: label})
        else:
            col_tokens, token_to_word_mappings, word_offsets = tokenize_ner_text(ner_text, self.tokenizer)
            ret.update({self.model.label_key: np.array([], dtype=np.int32)})

        ret.update(
            {
                self.model.text_token_ids_key: np.array(col_tokens.input_ids, dtype=np.int32),
                self.model.text_valid_length_key: sum(col_tokens.attention_mask),
                self.model.text_segment_ids_key: np.array(col_tokens.token_type_ids, dtype=np.int32),
                self.model.text_token_word_mapping_key: token_to_word_mappings,
                self.model.text_word_offsets_key: word_offsets,
            }
        )

        return ret

    def __call__(
        self,
        all_features: Dict[str, Union[NDArray[(Any,), Any], list]],
        feature_modalities: Dict[str, Union[NDArray[(Any,), Any], list]],
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
        ret = {}
        if self.prefix == NER_TEXT:
            ret = self.process_ner(all_features, feature_modalities, is_training)

        return ret
