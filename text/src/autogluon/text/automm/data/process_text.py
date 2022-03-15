import os
import logging
from typing import Optional, List, Any
import numpy as np
from nptyping import NDArray
import warnings
from transformers import (
    BertTokenizer,
    CLIPTokenizer,
    ElectraTokenizer,
    AutoTokenizer,
    AutoConfig,
)
from ..constants import (
    TEXT_TOKEN_IDS,
    TEXT_VALID_LENGTH,
    TEXT_SEGMENT_IDS,
    AUTOMM,
)
from .collator import Stack, Pad
from .utils import extract_value_from_config

logger = logging.getLogger(AUTOMM)

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


ALL_TOKENIZERS = {
    "bert": BertTokenizer,
    "clip": CLIPTokenizer,
    "electra": ElectraTokenizer,
    'hf_auto': AutoTokenizer,
}


class TextProcessor:
    """
    Prepare text data for the model specified by "prefix". For multiple models requiring text data,
    we need to create a TextProcessor for each related model so that they will have independent input.
    """

    def __init__(
            self,
            prefix: str,
            checkpoint_name: str,
            tokenizer_name: Optional[str] = "hf_auto",
            max_len: Optional[int] = None,
            insert_sep: Optional[bool] = True,
            text_segment_num: Optional[int] = 1,
            stochastic_chunk: Optional[bool] = False,
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
        """
        self.prefix = prefix
        self.tokenizer = self.get_pretrained_tokenizer(
            tokenizer_name=tokenizer_name,
            checkpoint_name=checkpoint_name,
        )
        if hasattr(self.tokenizer, 'deprecation_warnings'):
            # Disable the warning "Token indices sequence length is longer than the specified maximum sequence..."
            # See https://github.com/huggingface/transformers/blob/6ac77534bfe97c00e0127bb4fc846ae0faf1c9c5/src/transformers/tokenization_utils_base.py#L3362
            self.tokenizer.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True

        self.cls_token_id, self.sep_token_id = self.get_special_tokens(tokenizer=self.tokenizer)
        if max_len is None or max_len <= 0:
            self.max_len = self.tokenizer.model_max_length
        else:
            if max_len < self.tokenizer.model_max_length:
                warnings.warn(
                    f"provided max length: {max_len} "
                    f"is smaller than {checkpoint_name}'s default: {self.tokenizer.model_max_length}"
                )
            self.max_len = min(max_len, self.tokenizer.model_max_length)
        logger.debug(f"text max length: {self.max_len}")

        self.insert_sep = insert_sep

        config = AutoConfig.from_pretrained(checkpoint_name).to_diff_dict()
        extracted = extract_value_from_config(
            config=config,
            keys=("type_vocab_size",)
        )
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
                f"is smaller than {checkpoint_name}'s default: {default_segment_num}"
            )
        self.text_segment_num = min(text_segment_num, default_segment_num)
        assert self.text_segment_num >= 1
        logger.debug(f"text segment num: {self.text_segment_num}")

        self.stochastic_chunk = stochastic_chunk

    def collate_fn(self) -> dict:
        """
        Collate text features into a batch.
        This function will be used when creating Pytorch DataLoader.

        Returns
        -------
        A dictionary containing one model's collator function for text data.
        """
        fn = {}
        fn.update({f"{self.prefix}_{TEXT_TOKEN_IDS}": Pad(pad_val=self.tokenizer.pad_token_id)})
        fn.update({f"{self.prefix}_{TEXT_VALID_LENGTH}": Stack()})
        fn.update({f"{self.prefix}_{TEXT_SEGMENT_IDS}": Pad(pad_val=0)})
        return fn

    def build_one_token_sequence(
            self,
            text_tokens: List[NDArray[(Any,), np.int32]],
    ) -> dict:
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
        trimmed_lengths = self.get_trimmed_lengths(
            [len(txt_token) for txt_token in text_tokens],
            max_length,
            do_merge=True,
        )
        seg = 0
        token_ids = [self.cls_token_id]
        segment_ids = [seg]
        for txt_token, trim_length in zip(text_tokens, trimmed_lengths):

            if self.stochastic_chunk:
                start_ptr = np.random.randint(0, len(txt_token) - trim_length + 1)
            else:
                start_ptr = 0
            token_ids.extend(txt_token[start_ptr:(start_ptr + trim_length)].tolist())
            segment_ids.extend([seg] * trim_length)
            if self.insert_sep:
                token_ids.append(self.sep_token_id)
                segment_ids.append(seg)
            seg = (seg + 1) % self.text_segment_num

        if token_ids[-1] != self.sep_token_id:
            token_ids.append(self.sep_token_id)
            segment_ids.append(seg)

        return {
            f"{self.prefix}_{TEXT_TOKEN_IDS}": np.array(token_ids, dtype=np.int32),
            f"{self.prefix}_{TEXT_VALID_LENGTH}": len(token_ids),
            f"{self.prefix}_{TEXT_SEGMENT_IDS}": np.array(segment_ids, dtype=np.int32)
        }

    def build_one_token_sequence_from_text(
            self,
            text: List[str],
    ) -> dict:
        """
        Tokenize a sample's text data and build one token sequence. One sample may have
        multiple text columns in a multimodal pd.DataFrame.

        Parameters
        ----------
        text
            The raw text data of one sample.

        Returns
        -------
        A dictionary containing one sample's text tokens, valid length, and segment ids.
        """
        # tokenize text
        tokens = []
        warnings.filterwarnings(
            "ignore",
            "Token indices sequence length is longer than.*result in indexing errors"
        )
        for col_text in text:
            col_tokens = self.tokenizer.encode(
                col_text,
                add_special_tokens=False,
                truncation=False,
            )
            tokens.append(np.array(col_tokens, dtype=np.int32))

        # build token sequence
        return self.build_one_token_sequence(tokens)

    @staticmethod
    def get_special_tokens(tokenizer):
        """
        Extract the cls and sep token ids from a huggingface tokenizer. In most cases,
        we can use the attributes "cls_token_id" and "sep_token_id". But for CLIP, we
        need to use "bos_token_id" and "eos_token_id".

        Parameters
        ----------
        tokenizer
            A huggingface tokenizer instance.

        Returns
        -------
        The cls and sep token ids.
        """
        cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
        if cls_id is None or sep_id is None:
            cls_id, sep_id = tokenizer.bos_token_id, tokenizer.eos_token_id
        if cls_id is None or sep_id is None:
            raise ValueError(
                f"tokenizer class: {tokenizer.__class__.__name__} has no valid cls and sep ids."
            )
        return cls_id, sep_id

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
            all_text: List[List[str]],
            idx: int,
            is_training: bool,
    ) -> dict:
        """
        Extract one sample's text data, tokenize them, and build one token sequence.

        Parameters
        ----------
        all_text
            All the raw text data in a dataset.
        idx
            The sample index in a dataset.
        is_training
            Whether to do processing in the training mode. This unused flag is for the API compatibility.

        Returns
        -------
        A dictionary containing one sample's text tokens, valid length, and segment ids.
        """
        per_sample_text = [per_column_text[idx] for per_column_text in all_text]
        return self.build_one_token_sequence_from_text(per_sample_text)
