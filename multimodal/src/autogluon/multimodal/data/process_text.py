import ast
import codecs
import logging
import os
import random
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig
from text_unidecode import unidecode
from torch import nn

from ..constants import CHOICES_IDS, COLUMN, TEXT, TEXT_SEGMENT_IDS, TEXT_TOKEN_IDS, TEXT_VALID_LENGTH
from ..models.utils import get_pretrained_tokenizer
from .collator import PadCollator, StackCollator
from .template_engine import TemplateEngine
from .trivial_augmenter import TrivialAugment

logger = logging.getLogger(__name__)

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TextProcessor:
    """
    Prepare text data for the model specified by "prefix". For multiple models requiring text data,
    we need to create a TextProcessor for each related model so that they will have independent input.
    """

    def __init__(
        self,
        model: nn.Module,
        insert_sep: Optional[bool] = True,
        stochastic_chunk: Optional[bool] = False,
        requires_column_info: bool = False,
        text_detection_length: Optional[int] = None,
        text_trivial_aug_maxscale: Optional[float] = 0.0,
        train_augment_types: Optional[List[str]] = None,
        template_config: Optional[DictConfig] = None,
        normalize_text: Optional[bool] = False,
        dropout: Optional[float] = 0,
    ):
        """
        Parameters
        ----------
        model
            The model for which this processor would be created.
        insert_sep
            Whether to insert SEP tokens.
        stochastic_chunk
            Whether to use stochastic chunking, which will randomly slice each individual text.
        requires_column_info
            Whether to require feature column information in dataloader.
        text_detection_length
            A naive way to detect text column versus tabular column that were treated as text
        text_trivial_aug_maxscale
            Used in trivial augment as the maximum scale that can be random generated
            A value of 0 means turn off trivial augment
            https://arxiv.org/pdf/2103.10158.pdf
        train_augment_types
            All possible augmentation operations
        normalize_text
            Whether to normalize text to resolve encoding problems.
            Examples of normalized texts can be found at
            https://github.com/autogluon/autogluon/tree/master/examples/automm/kaggle_feedback_prize#15-a-few-examples-of-normalized-texts
        """
        logger.debug(f"initializing text processor for model {model.prefix}")
        self.prefix = model.prefix
        self.requires_column_info = requires_column_info
        self.tokenizer_name = model.tokenizer_name
        # model should have a tokenizer
        self.tokenizer = model.tokenizer
        if hasattr(self.tokenizer, "deprecation_warnings"):
            # Disable the warning "Token indices sequence length is longer than the specified maximum sequence..."
            # See https://github.com/huggingface/transformers/blob/6ac77534bfe97c00e0127bb4fc846ae0faf1c9c5/src/transformers/tokenization_utils_base.py#L3362
            self.tokenizer.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True

        self.cls_token_id, self.sep_token_id, self.eos_token_id = self.get_special_tokens(tokenizer=self.tokenizer)
        self.max_len = model.max_text_len
        self.insert_sep = insert_sep
        self.eos_only = self.cls_token_id == self.sep_token_id == self.eos_token_id
        self.text_segment_num = model.text_segment_num

        self.stochastic_chunk = stochastic_chunk
        self.normalize_text = normalize_text
        assert 0 <= dropout <= 1
        if dropout > 0:
            logger.debug(f"text dropout probability: {dropout}")
        self.dropout = dropout

        # construct augmentor
        self.train_augment_types = train_augment_types
        self.text_detection_length = text_detection_length
        self.text_trivial_aug_maxscale = text_trivial_aug_maxscale
        self.train_augmenter = self.construct_text_augmenter(self.text_trivial_aug_maxscale, self.train_augment_types)
        self.template_config = template_config
        if self.template_config.turn_on:
            self.template_engine = TemplateEngine(self.template_config)
        else:
            self.template_engine = None

        if self.normalize_text:
            self.register_encoding_decoding_error_handlers()

    @property
    def text_token_ids_key(self):
        return f"{self.prefix}_{TEXT_TOKEN_IDS}"

    @property
    def text_segment_ids_key(self):
        return f"{self.prefix}_{TEXT_SEGMENT_IDS}"

    @property
    def choices_ids_key(self):
        return f"{self.prefix}_{CHOICES_IDS}"

    @property
    def text_valid_length_key(self):
        return f"{self.prefix}_{TEXT_VALID_LENGTH}"

    @property
    def text_column_prefix(self):
        return f"{self.text_token_ids_key}_{COLUMN}"

    def collate_fn(self, text_column_names: Optional[List] = None) -> Dict:
        """
        Collate text features into a batch.
        This function will be used when creating Pytorch DataLoader.

        Returns
        -------
        A dictionary containing one model's collator function for text data.
        """
        fn = {}
        if self.requires_column_info:
            assert text_column_names, "Empty text column names."
            for col_name in text_column_names:
                fn[f"{self.text_column_prefix}_{col_name}"] = StackCollator()

        fn.update(
            {
                self.text_token_ids_key: PadCollator(pad_val=self.tokenizer.pad_token_id),
                self.text_valid_length_key: StackCollator(),
                self.text_segment_ids_key: PadCollator(pad_val=0),
                self.choices_ids_key: PadCollator(pad_val=0),
            }
        )

        return fn

    def build_one_token_sequence(
        self,
        text_tokens: Dict[str, NDArray],
    ) -> Dict:
        """
        Construct one token sequence based on multiple token sequences coming from different
        text columns in a multimodal pd.DataFrame. The token sequence length and the text segment
        id are upper bounded by "self.max_len" and "self.text_segment_num".

        Parameters
        ----------
        text_tokens
            One sample's text token sequences from different text columns in a multimodal pd.DataFrame.

        Returns
        -------
        A dictionary containing one sample's text tokens, valid length, and segment ids.
        """
        if self.insert_sep:
            max_length = self.max_len - (len(text_tokens) + 1)
        else:
            max_length = self.max_len - 2
        if self.eos_only:
            # For EOS-only, the tokens will be combined as
            # [Field1 Tokens] [EOS] [Field2 Tokens] [EOS] [Field3 Tokens] [EOS]
            # Otherwise, the tokens will be combined as
            # [CLS] [Field1 Tokens] [SEP] [Field2 Tokens] [SEP] [Field3 Tokens] [EOS]
            max_length += 1
        trimmed_lengths = self.get_trimmed_lengths(
            [len(txt_token) for txt_token in text_tokens.values()],
            max_length,
            do_merge=True,
        )
        seg = 0
        if self.eos_only:
            # There is no cls token in the EOS-only mode
            token_ids = []
        else:
            token_ids = [self.cls_token_id]

        choices_ids = []
        segment_ids = [seg]
        ret = {}

        for (col_name, txt_token), trim_length in zip(text_tokens.items(), trimmed_lengths):
            if col_name == CHOICES_IDS:
                choices_ids = txt_token
                continue
            segment_start = len(token_ids)
            if self.stochastic_chunk:
                start_ptr = np.random.randint(0, len(txt_token) - trim_length + 1)
            else:
                start_ptr = 0
            token_ids.extend(txt_token[start_ptr : (start_ptr + trim_length)].tolist())
            segment_ids.extend([seg] * trim_length)
            if self.requires_column_info:
                # np.int64 corresponds to torch.LongTensor
                col_token_idxs = np.array([segment_start, segment_start + trim_length], dtype=np.int64)
                ret[f"{self.text_column_prefix}_{col_name}"] = col_token_idxs
            if self.insert_sep:
                token_ids.append(self.sep_token_id)
                segment_ids.append(seg)
            seg = (seg + 1) % self.text_segment_num

        if token_ids[-1] != self.eos_token_id:
            token_ids.append(self.eos_token_id)
            segment_ids.append(seg)

        ret.update(
            {
                self.text_token_ids_key: np.array(token_ids, dtype=np.int32),
                self.text_valid_length_key: len(token_ids),
                self.text_segment_ids_key: np.array(segment_ids, dtype=np.int32),
                self.choices_ids_key: np.array(choices_ids, dtype=np.int32),
            }
        )

        return ret

    def build_one_token_sequence_from_text(
        self,
        text: Dict[str, str],
        is_training: bool,
    ) -> Dict:
        """
        Tokenize a sample's text data and build one token sequence. One sample may have
        multiple text columns in a multimodal pd.DataFrame.

        Parameters
        ----------
        text
            The raw text data of one sample.

        is_training
            Flag to apply augmentation only to training.

        Returns
        -------
        A dictionary containing one sample's text tokens, valid length, and segment ids.
        """
        # tokenize text
        tokens = {}
        warnings.filterwarnings("ignore", "Token indices sequence length is longer than.*result in indexing errors")

        if self.template_config.turn_on:
            template, applied_template = self.template_engine.sample_and_apply_template(text)
            if template:
                answer_choices = template.get_answer_choices_list(text)
                text = {}
                text[TEXT_TOKEN_IDS] = applied_template[0]
                text[CHOICES_IDS] = answer_choices

        for col_name, col_text in text.items():
            if is_training:
                if self.dropout > 0 and random.uniform(0, 1) <= self.dropout:
                    col_text = ""
                elif self.train_augmenter is not None:
                    # naive way to detect categorical/numerical text:
                    if len(col_text.split(" ")) >= self.text_detection_length:
                        col_text = self.train_augmenter(col_text)
                        # After text augmentation, "col_text" may become a list. An error will be raised when calling "tokenizer.encode".
                        if type(col_text) == list and len(col_text) == 1:
                            col_text = col_text[0]

            if col_name == CHOICES_IDS:
                answer_ids = self.tokenizer(
                    col_text,
                    padding="max_length",
                    max_length=self.template_engine.get_max_choice_length(self.tokenizer),
                )["input_ids"]
                tokens[col_name] = answer_ids
                continue
            col_tokens = self.tokenizer.encode(
                col_text,
                add_special_tokens=False,
                truncation=False,
            )
            tokens[col_name] = np.array(col_tokens, dtype=np.int32)
        # build token sequence
        return self.build_one_token_sequence(tokens)

    @staticmethod
    def get_special_tokens(tokenizer):
        """
        Extract the cls, sep, and eos token ids from a huggingface tokenizer. In most cases,
        we can use the attributes "cls_token_id" and "sep_token_id". But for CLIP, we
        need to use "bos_token_id" and "eos_token_id".

        Parameters
        ----------
        tokenizer
            A huggingface tokenizer instance.

        Returns
        -------
        The cls, sep, and eos token ids.
        """
        cls_id, sep_id, eos_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.sep_token_id
        if cls_id is None or sep_id is None:
            # CLIP uses eos_token's feature as the pooled output.
            # See https://github.com/huggingface/transformers/blob/v4.14.1/src/transformers/models/clip/modeling_clip.py#L657
            cls_id, sep_id, eos_id = tokenizer.bos_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id

        if cls_id is None and sep_id is None:
            # Special treatment for T5 (EOS-only).
            cls_id = sep_id = eos_id

        if cls_id is None or sep_id is None or eos_id is None:
            raise ValueError(f"tokenizer class: {tokenizer.__class__.__name__} has no valid cls, sep, and eos ids.")
        return cls_id, sep_id, eos_id

    @staticmethod
    def get_trimmed_lengths(
        lengths: List[int],
        max_length: int,
        do_merge: bool = False,
    ) -> List:
        """
        Get the trimmed lengths of multiple text token sequences. It will make sure that
        the trimmed length is smaller than or equal to the max_length.
        - do_merge is True
            Make sure that sum(trimmed_lengths) <= max_length.
            The strategy is always trying to trim the longer lengths.
        - do_merge is False
            Make sure that all(trimmed_lengths <= max_length).

        Parameters
        ----------
        lengths
            The original lengths of each token sequence.
        max_length
            When do_merge is True,
                We set the max_length constraint on the total length.
            When do_merge is False,
                We set the max_length constraint on individual sequences.
        do_merge
            Whether these sentences will be merged.

        Returns
        -------
        trimmed_lengths
            The trimmed lengths of each individual text field.
        """
        lengths = np.array(lengths)
        if do_merge:
            total_length = sum(lengths)
            if total_length <= max_length:
                return list(lengths)
            trimmed_lengths = np.zeros_like(lengths)
            while sum(trimmed_lengths) != max_length:
                remainder = max_length - sum(trimmed_lengths)
                budgets = lengths - trimmed_lengths
                nonzero_idx = (budgets > 0).nonzero()[0]
                nonzero_budgets = budgets[nonzero_idx]
                if remainder < len(nonzero_idx):
                    for i in range(remainder):
                        trimmed_lengths[nonzero_idx[i]] += 1
                else:
                    increment = min(min(nonzero_budgets), remainder // len(nonzero_idx))
                    trimmed_lengths[nonzero_idx] += increment
            return list(trimmed_lengths)
        else:
            return list(np.minimum(lengths, max_length))

    @staticmethod
    def construct_text_augmenter(
        augment_maxscale: float,
        augment_types: List[str],
    ) -> Optional[TrivialAugment]:
        """
        Build up a text augmentor from the provided list of augmentation types

        Parameters
        ----------
        augment_maxscale:
            maximum scale for text augmentation
        augment_types
            A list of text augment types.

        Returns
        -------
        A trivial augment instance.
        """
        if augment_maxscale == 0.0 or augment_maxscale is None:
            return None

        if augment_types is None or len(augment_types) == 0:
            return TrivialAugment(TEXT, max_strength=augment_maxscale)
        else:
            auglist = []
            for aug_type in augment_types:
                if "(" in aug_type:
                    trans_mode = aug_type[0 : aug_type.find("(")]
                    args = ast.literal_eval(aug_type[aug_type.find("(") :])
                else:
                    trans_mode = aug_type
                    args = None

                auglist.append((trans_mode, args))

            return TrivialAugment(TEXT, augment_maxscale, auglist)

    def __call__(
        self,
        text: Dict[str, str],
        sub_dtypes: Dict[str, str],
        is_training: bool,
    ) -> Dict:
        """
        Extract one sample's text data, tokenize them, and build one token sequence.

        Parameters
        ----------
        text
            Text of one sample.
        sub_dtypes
            The sub data types of all text columns.
        is_training
            Whether to do processing in the training mode.

        Returns
        -------
        A dictionary containing one sample's text tokens, valid length, and segment ids.
        """
        if self.normalize_text:
            text = {col_name: self.normalize_txt(col_text) for col_name, col_text in text.items()}

        return self.build_one_token_sequence_from_text(text=text, is_training=is_training)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k != "train_augmenter":
                setattr(result, k, deepcopy(v, memo))
        # manual reconstruct augmenter
        result.train_augmenter = self.construct_text_augmenter(
            result.text_trivial_aug_maxscale, result.train_augment_types
        )
        return result

    def __getstate__(self):
        odict = self.__dict__.copy()  # get attribute dictionary
        del odict["train_augmenter"]  # remove textaugmenter to support pickle
        return odict

    def __setstate__(self, state):
        self.__dict__ = state
        self.train_augmenter = self.construct_text_augmenter(
            state["text_trivial_aug_maxscale"], state["train_augment_types"]
        )

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

    @staticmethod
    def normalize_txt(text: str) -> str:
        """Resolve the encoding problems and normalize the abnormal characters."""

        text = (
            text.encode("raw_unicode_escape")
            .decode("utf-8", errors="replace_decoding_with_cp1252")
            .encode("cp1252", errors="replace_encoding_with_utf8")
            .decode("utf-8", errors="replace_decoding_with_cp1252")
        )
        text = unidecode(text)
        return text

    @staticmethod
    def register_encoding_decoding_error_handlers() -> None:
        """Register the encoding and decoding error handlers for `utf-8` and `cp1252`."""

        def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
            return error.object[error.start : error.end].encode("utf-8"), error.end

        def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
            return error.object[error.start : error.end].decode("cp1252"), error.end

        codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
        codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)
