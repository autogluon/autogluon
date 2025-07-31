import logging
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig
from tokenizers import pre_tokenizers
from torch import nn

from ..constants import NER_ANNOTATION, NER_TEXT, TEXT, TEXT_NER
from ..models.utils import get_pretrained_tokenizer
from .collator import PadCollator, StackCollator

logger = logging.getLogger(__name__)


class NerProcessor:
    """
    Prepare NER data for the model specified by "prefix".
    """

    def __init__(
        self,
        model: nn.Module,
        max_len: Optional[int] = None,
        entity_map: Optional[DictConfig] = None,
    ):
        """
        Parameters
        ----------
        model
            The NER model.
        max_len
            The max length of the tokenizer.
        entity_map
            The map between tags and tag indexes. e.g., {"PER":2, "LOC":3}.
        """
        self.prefix = model.prefix
        self.text_token_ids_key = model.text_token_ids_key
        self.text_valid_length_key = model.text_valid_length_key
        self.text_segment_ids_key = model.text_segment_ids_key
        self.text_token_word_mapping_key = model.text_token_word_mapping_key
        self.text_word_offsets_key = model.text_word_offsets_key
        self.label_key = model.label_key

        self.tokenizer = None
        self.tokenizer_name = model.tokenizer_name
        self.max_len = max_len
        self.entity_map = entity_map

        # Disable tokenizer parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
                    self.text_token_ids_key: PadCollator(pad_val=self.tokenizer.pad_token_id),
                    self.text_valid_length_key: StackCollator(),
                    self.text_segment_ids_key: PadCollator(pad_val=0),
                    self.text_token_word_mapping_key: PadCollator(pad_val=0),
                    self.text_word_offsets_key: PadCollator(pad_val=0),
                    self.label_key: StackCollator(),
                }
            )
        return fn

    def process_ner(
        self,
        all_features: Dict[str, Union[int, float, list]],
        data_types: Dict[str, Union[int, float, list]],
        is_training: bool,
    ) -> Dict:
        """
        Process one NER sample's label and text features.
        New rules can be added if necessary.

        Parameters
        ----------
        all_features
            All features including text and ner annotations.
        data_types
            Data type of all columns.

        Returns
        -------
        A dictionary containing one NER sample's features and/or labels.
        """
        ret = {}
        # overwrite model_max_length for standalone checkpoints if it's not specified.
        if self.max_len is not None and self.tokenizer.model_max_length > 10**6:
            self.tokenizer.model_max_length = self.max_len
        text_column, annotation_column = None, None
        for column_name, column_modality in data_types.items():
            if column_modality.startswith((TEXT_NER, TEXT)):
                text_column = column_name
            if column_modality == NER_ANNOTATION:
                annotation_column = column_name

        ner_text = all_features[text_column]
        if is_training or annotation_column is not None:
            ner_annotation = all_features[annotation_column]
            label, col_tokens, token_to_word_mappings, word_offsets = self.process_ner_annotations(
                ner_annotation, ner_text, self.entity_map, self.tokenizer
            )
            ret.update({self.label_key: label})
        else:
            col_tokens, token_to_word_mappings, word_offsets = self.tokenize_ner_text(ner_text, self.tokenizer)
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

    @classmethod
    def process_ner_annotations(cls, ner_annotations, ner_text, entity_map, tokenizer, is_eval=False):
        """
        Generate token-level/word-level labels with given text and NER annotations.

        Parameters
        ----------
        ner_annotations
            The NER annotations.
        ner_text
            The corresponding raw text.
        entity_map
            The map between tags and tag indexes. e.g., {"PER":2, "LOC":3}.
        tokenizer
            The tokenizer to be used.
        is_eval
            Whether it is for evaluation or not, default: False

        Returns
        -------
        Token-level/word-level labels and text features.
        """
        col_tokens, token_to_word_mappings, word_offsets = cls.tokenize_ner_text(ner_text, tokenizer)
        num_words = len(set(token_to_word_mappings)) - 1
        word_label = [1] * num_words
        # TODO: Potentially optimize word label generation via binary search
        b_prefix = "B-"
        i_prefix = "I-"
        for annot in ner_annotations:
            custom_offset = annot[0]
            custom_label = annot[1]
            is_start_word = True
            for idx, word_offset in enumerate(word_offsets[:num_words, :]):
                # support multiple words in an annotated offset range.
                # Allow partial overlapping between custom annotations and pretokenized words.
                if (word_offset[0] < custom_offset[1]) and (custom_offset[0] < word_offset[1]):
                    if not (
                        re.match(b_prefix, custom_label, re.IGNORECASE)
                        or re.match(i_prefix, custom_label, re.IGNORECASE)
                    ):
                        if is_start_word and b_prefix + custom_label in entity_map:
                            word_label[idx] = entity_map[b_prefix + custom_label]
                            is_start_word = False
                        elif i_prefix + custom_label in entity_map:
                            word_label[idx] = entity_map[i_prefix + custom_label]
                    else:
                        if custom_label in entity_map:
                            word_label[idx] = entity_map[custom_label]

        token_label = [0] * len(col_tokens.input_ids)
        temp = set()
        counter = 0
        for idx, token_to_word in enumerate(token_to_word_mappings):
            if token_to_word != -1 and token_to_word not in temp:
                temp.add(token_to_word)
                token_label[idx] = word_label[counter]
                counter += 1
        if not is_eval:
            label = token_label  # return token-level labels for training
        else:
            label = word_label  # return word-level labels for evaluation

        return label, col_tokens, token_to_word_mappings, word_offsets

    @classmethod
    def tokenize_ner_text(cls, text, tokenizer):
        """
        Tokenization process for the NER task. It will be used for the token-level label generation
        and the input text tokenization.

        Parameters
        ----------
        text
            The raw text data.
        tokenizer
            The tokenizer to be used.

        Returns
        -------
        The output of tokenizer and word offsets.
        """
        # pre-tokenization is required for NER token-level label generation.
        words_with_offsets = pre_tokenizers.BertPreTokenizer().pre_tokenize_str(text)
        words_with_offsets = (
            cls.is_space_counted(words_with_offsets) if len(words_with_offsets) > 1 else words_with_offsets
        )
        words = [word for word, offset in words_with_offsets]
        word_offsets = np.array([[offset[0], offset[1]] for word, offset in words_with_offsets], dtype=np.int32)
        col_tokens = tokenizer(
            words,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_token_type_ids=True,
        )
        offset_mapping = np.array(col_tokens.offset_mapping, dtype=np.int32)
        if len(words_with_offsets) > 1:
            if offset_mapping.shape[0] > len(words):
                word_offsets = np.pad(word_offsets, ((0, offset_mapping.shape[0] - len(words)), (0, 0)), "constant")
            # token to word mappings: it will tell us which token belongs to which word.
            token_to_word_mappings = [i if i != None else -1 for i in col_tokens.word_ids()]
            if len(set(token_to_word_mappings)) != len(words) + 1:
                warnings.warn(f"The token to word mappings are incorrect!")
        else:
            # If pre_tokenizer does not give word offsets, use word_ids and offset_mappings instead.
            word_offsets = np.append(offset_mapping[1:], [[0, 0]], axis=0)
            word_idx = np.arange(len(col_tokens.word_ids()) - col_tokens.word_ids().count(None))
            token_to_word_mappings = [
                val + word_idx[idx - 1] if val != None else -1 for idx, val in enumerate(col_tokens.word_ids())
            ]

        return col_tokens, token_to_word_mappings, word_offsets

    @staticmethod
    def is_space_counted(words_with_offsets):
        """
        Some tokenizers will count space into words for example.
        Given text: 'hello world', normal bert will output: [('hello', (0, 5)), ('world', (6, 11))]
        while some checkpoint will output: [('▁hello', (0, 5)), ('▁world', (5, 11))]
        This will lead to inconsistency issue during labelling, details can be found here:
        https://github.com/huggingface/transformers/issues/18111

        This function will check whether space is counted or not and realign the offset.
        """
        offset0, offset1 = [], []
        for word, offset in words_with_offsets:
            offset0.append(offset[0])
            offset1.append(offset[1])

        realign = []
        if offset0[1:] == offset1[:-1]:  # space are counted
            realign = [words_with_offsets[0]]
            for word, offset in words_with_offsets[1:]:
                if word.startswith("▁"):  # it is "Lower One Eighth Block" (U+2581) rather than lower line (U+005F).
                    realign.append((word, (offset[0] + 1, offset[1])))
                else:
                    realign.append((word, offset))

        if realign:
            return realign
        else:
            return words_with_offsets

    def save_tokenizer(
        self,
        path: str,
    ):
        """
        Save the text tokenizer and record its relative paths, e.g, hf_text.

        Parameters
        ----------
        path
            The root path of saving.

        """
        save_path = os.path.join(path, self.prefix)
        self.tokenizer.save_pretrained(save_path)
        self.tokenizer = self.prefix

    def load_tokenizer(
        self,
        path: str,
    ):
        """
        Load saved text tokenizers. If text/ner processors already have tokenizers,
        then do nothing.

        Parameters
        ----------
        path
            The root path of loading.

        Returns
        -------
        A list of text/ner processors with tokenizers loaded.
        """
        if isinstance(self.tokenizer, str):
            load_path = os.path.join(path, self.tokenizer)
            self.tokenizer = get_pretrained_tokenizer(
                tokenizer_name=self.tokenizer_name,
                checkpoint_name=load_path,
            )

    def __call__(
        self,
        all_features: Dict[str, Union[NDArray, list]],
        data_types: Dict[str, Union[NDArray, list]],
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
            ret = self.process_ner(all_features, data_types, is_training)

        return ret
