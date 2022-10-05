import ast
import logging
import os
import warnings
from copy import deepcopy
from lib2to3.pgen2.token import OP
from typing import Any, Dict, List, Optional

import numpy as np
from nptyping import NDArray
from omegaconf import DictConfig
from torch import nn
from transformers import AutoConfig, AutoTokenizer, BertTokenizer, CLIPTokenizer, ElectraTokenizer

from ..constants import AUTOMM, CHOICES_IDS, COLUMN, TEXT, TEXT_SEGMENT_IDS, TEXT_TOKEN_IDS, TEXT_VALID_LENGTH
from .collator import Pad, Stack
from .template_engine import TemplateEngine
from .trivial_augmenter import TrivialAugment
from .utils import extract_value_from_config, normalize_txt, register_encoding_decoding_error_handlers

logger = logging.getLogger(AUTOMM)

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


ALL_TOKENIZERS = {
    "bert": BertTokenizer,
    "clip": CLIPTokenizer,
    "electra": ElectraTokenizer,
    "hf_auto": AutoTokenizer,
}


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


class TextProcessor:
    """
    Prepare text data for the model specified by "prefix". For multiple models requiring text data,
    we need to create a TextProcessor for each related model so that they will have independent input.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer_name: Optional[str] = "hf_auto",
        max_len: Optional[int] = None,
        insert_sep: Optional[bool] = True,
        text_segment_num: Optional[int] = 1,
        stochastic_chunk: Optional[bool] = False,
        requires_column_info: bool = False,
        text_detection_length: Optional[int] = None,
        text_trivial_aug_maxscale: Optional[float] = 0.0,
        train_augment_types: Optional[List[str]] = None,
        template_config: Optional[DictConfig] = None,
        normalize_text: Optional[bool] = False,
    ):
        """
        Parameters
        ----------
        prefix
            The prefix connecting a processor to its corresponding model.
        checkpoint_name
            Name of the pretrained huggingface checkpoint, e.g., "microsoft/deberta-v3-small"
        tokenizer_name
            Name of the huggingface tokenizer type (default "hf_auto").
        max_len
            The maximum length of text tokens.
        insert_sep
            Whether to insert SEP tokens.
        text_segment_num
            The number of text segments.
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
            https://github.com/awslabs/autogluon/tree/master/examples/automm/kaggle_feedback_prize#15-a-few-examples-of-normalized-texts
        """
        self.prefix = model.prefix
        self.tokenizer_name = tokenizer_name
        self.requires_column_info = requires_column_info
        self.tokenizer = self.get_pretrained_tokenizer(
            tokenizer_name=tokenizer_name,
            checkpoint_name=model.checkpoint_name,
        )
        if hasattr(self.tokenizer, "deprecation_warnings"):
            # Disable the warning "Token indices sequence length is longer than the specified maximum sequence..."
            # See https://github.com/huggingface/transformers/blob/6ac77534bfe97c00e0127bb4fc846ae0faf1c9c5/src/transformers/tokenization_utils_base.py#L3362
            self.tokenizer.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True

        self.cls_token_id, self.sep_token_id, self.eos_token_id = self.get_special_tokens(tokenizer=self.tokenizer)
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

        self.insert_sep = insert_sep
        self.eos_only = self.cls_token_id == self.sep_token_id == self.eos_token_id

        extracted = extract_value_from_config(config=model.config.to_diff_dict(), keys=("type_vocab_size",))
        if len(extracted) == 0:
            default_segment_num = 1
        elif len(extracted) == 1:
            default_segment_num = extracted[0]
        else:
            raise ValueError(f" more than one type_vocab_size values are detected: {extracted}")

        if default_segment_num <= 0:
            default_segment_num = 1

        if text_segment_num < default_segment_num:
            warnings.warn(
                f"provided text_segment_num: {text_segment_num} "
                f"is smaller than {model.checkpoint_name}'s default: {default_segment_num}"
            )
        self.text_segment_num = min(text_segment_num, default_segment_num)
        assert self.text_segment_num >= 1
        logger.debug(f"text segment num: {self.text_segment_num}")

        self.stochastic_chunk = stochastic_chunk
        self.normalize_text = normalize_text

        # construct augmentor
        self.train_augment_types = train_augment_types
        self.text_detection_length = text_detection_length
        self.text_trivial_aug_maxscale = text_trivial_aug_maxscale
        self.train_augmenter = construct_text_augmenter(self.text_trivial_aug_maxscale, self.train_augment_types)
        self.template_config = template_config
        self.template_engine = TemplateEngine(self.template_config)

        if self.normalize_text:
            register_encoding_decoding_error_handlers()

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
                fn[f"{self.text_column_prefix}_{col_name}"] = Stack()

        fn.update(
            {
                self.text_token_ids_key: Pad(pad_val=self.tokenizer.pad_token_id),
                self.text_valid_length_key: Stack(),
                self.text_segment_ids_key: Pad(pad_val=0),
                self.choices_ids_key: Pad(pad_val=0),
            }
        )

        return fn

    def build_one_token_sequence(
        self,
        text_tokens: Dict[str, NDArray[(Any,), np.int32]],
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

        if hasattr(self, "eos_token_id"):
            if token_ids[-1] != self.eos_token_id:
                token_ids.append(self.eos_token_id)
                segment_ids.append(seg)
        else:  # backward compatibility
            if token_ids[-1] != self.sep_token_id:
                token_ids.append(self.sep_token_id)
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
                if self.train_augmenter is not None:
                    # naive way to detect categorical/numerical text:
                    if len(col_text.split(" ")) >= self.text_detection_length:
                        col_text = self.train_augmenter(col_text)
            if col_name == CHOICES_IDS:
                answer_ids = self.tokenizer(
                    col_text,
                    return_tensors="pt",
                    padding=True,
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

    @staticmethod
    def get_trimmed_lengths(
        lengths: List[int],
        max_length: int,
        do_merge: bool = False,
    ) -> np.ndarray:
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
                return lengths
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
            return trimmed_lengths
        else:
            return np.minimum(lengths, max_length)

    def __call__(
        self,
        all_text: Dict[str, List[str]],
        idx: int,
        is_training: bool,
    ) -> Dict:
        """
        Extract one sample's text data, tokenize them, and build one token sequence.

        Parameters
        ----------
        all_text
            All the raw text data in a dataset.
        idx
            The sample index in a dataset.
        is_training
            Whether to do processing in the training mode.

        Returns
        -------
        A dictionary containing one sample's text tokens, valid length, and segment ids.
        """

        if self.normalize_text:
            per_sample_text = {
                per_column_name: normalize_txt(per_column_text[idx])
                for per_column_name, per_column_text in all_text.items()
            }
        else:
            per_sample_text = {
                per_column_name: per_column_text[idx] for per_column_name, per_column_text in all_text.items()
            }
        return self.build_one_token_sequence_from_text(per_sample_text, is_training)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k != "train_augmenter":
                setattr(result, k, deepcopy(v, memo))
        # manual reconstruct augmenter
        result.train_augmenter = construct_text_augmenter(result.text_trivial_aug_maxscale, result.train_augment_types)
        return result

    def __getstate__(self):
        odict = self.__dict__.copy()  # get attribute dictionary
        del odict["train_augmenter"]  # remove textaugmenter to support pickle
        return odict

    def __setstate__(self, state):
        self.__dict__ = state
        self.train_augmenter = construct_text_augmenter(
            state["text_trivial_aug_maxscale"], state["train_augment_types"]
        )
