import codecs
import random
import warnings
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from nlpaug import Augmenter
from nlpaug.util import Method
from text_unidecode import unidecode

from ..constants import IDENTIFIER, MMLAB_MODELS
from .collator import DictCollator
from .preprocess_dataframe import MultiModalFeaturePreprocessor


def extract_value_from_config(
    config: Dict,
    keys: Tuple[str, ...],
):
    """
    Traverse a config dictionary to get some hyper-parameter's value.

    Parameters
    ----------
    config
        A config dictionary.
    keys
        The possible names of a hyper-parameter.

    Returns
    -------
    The hyper-parameter value.
    """
    result = []
    for k, v in config.items():
        if k in keys:
            result.append(v)
        elif isinstance(v, dict):
            result += extract_value_from_config(v, keys)
        else:
            pass

    return result


class InsertPunctuation(Augmenter):
    """
    Inherit nlpaug basic augmenter to support insert random punction at random location https://arxiv.org/pdf/2108.13230.pdf

    example:
    a healthy ,clean , sweet little girl in Mantin . send me message if you can give her a nice home
    ? a ! healthy ,clean , sweet little : girl , in Mantin . send me message . if you ; can give her ? a nice home
    """

    def __init__(
        self,
        name="Insert_Punc",
        aug_min=1,
        aug_max=50,
        aug_p=0.3,
    ):
        """
        Parameters
        ----------
        name
            name used when print out augmentation function
        aug_min
            minimum number of punctuation to insert
        aug_max
            maximum number of punctuation to insert
        aug_p
            how many punctuation to insert calculated as aug_p * sentence length
        """
        super().__init__(
            name=name,
            method=Method.WORD,
            action="insert",
            aug_min=aug_min,
            aug_max=aug_max,
            aug_p=aug_p,
            device="cpu",
            include_detail=False,
            verbose=0,
        )
        self.punc_list = [".", ",", "!", "?", ";", ":"]

    def insert(self, data):
        """
        Random insert random punctuation at random location https://arxiv.org/pdf/2108.13230.pdf

        Parameters
        --------
        data: text


        Returns
        --------
        The augmented text

        """
        words = data.split(" ")
        cnt = random.randint(1, int(self.aug_p * len(words)))
        loc = random.sample(range(0, len(words)), cnt)
        new = []

        for i, word in enumerate(words):
            if i in loc:
                new.append(self.punc_list[random.randint(0, len(self.punc_list) - 1)])
                new.append(word)
            else:
                new.append(word)

        new = " ".join(new)
        return new

    @classmethod
    def clean(cls, data):
        if isinstance(data, list):
            return [d.strip() if d else d for d in data]
        return data.strip()

    @classmethod
    def is_duplicate(cls, dataset, data):
        for d in dataset:
            if d == data:
                return True
        return False


def get_collate_fn(
    df_preprocessor: Union[MultiModalFeaturePreprocessor, List[MultiModalFeaturePreprocessor]],
    data_processors: Union[Dict, List[Dict]],
    per_gpu_batch_size: Optional[int] = None,
):
    """
    Collect collator functions for each modality input of every model.
    These collator functions are wrapped by the "Dict" collator function,
    which can then be used by the Pytorch DataLoader.

    Parameters
    ----------
    df_preprocessor
        One or a list of dataframe preprocessors.
    data_processors
        One or a list of data processor dicts.
    per_gpu_batch_size
        Mini-batch size for each GPU.

    Returns
    -------
    A "Dict" collator wrapping other collators.
    """
    if isinstance(df_preprocessor, MultiModalFeaturePreprocessor):
        df_preprocessor = [df_preprocessor]
    if isinstance(data_processors, dict):
        data_processors = [data_processors]

    collate_fn = {}
    for per_preprocessor, per_data_processors_group in zip(df_preprocessor, data_processors):
        for per_modality in per_data_processors_group:
            per_modality_column_names = per_preprocessor.get_column_names(modality=per_modality)
            if per_modality_column_names:
                for per_model_processor in per_data_processors_group[per_modality]:
                    if per_model_processor.prefix.lower().startswith(MMLAB_MODELS):
                        collate_fn.update(
                            per_model_processor.collate_fn(
                                per_modality_column_names, per_gpu_batch_size=per_gpu_batch_size
                            )
                        )
                    else:
                        collate_fn.update(per_model_processor.collate_fn(per_modality_column_names))
    return DictCollator(collate_fn)


def apply_df_preprocessor(
    data: pd.DataFrame,
    df_preprocessor: MultiModalFeaturePreprocessor,
    modalities: Iterable,
):
    """
    Preprocess one dataframe with one df_preprocessor.

    Parameters
    ----------
    data
        A pandas dataframe.
    df_preprocessor
        One dataframe preprocessor object.
    modalities
        A list of data modalities to preprocess.

    Returns
    -------
    modality_features
        Preprocessed features of given modalities.
    modality_types
        Minor modality types of each major modality.
    sample_num
        Number of samples.
    """
    lengths = []
    modality_features = {}
    modality_types = {}
    for per_modality in modalities:
        per_modality_features, per_modality_types = getattr(df_preprocessor, f"transform_{per_modality}")(data)
        modality_features[per_modality] = per_modality_features
        modality_types[per_modality] = per_modality_types
        if per_modality_features:
            lengths.append(len(per_modality_features[next(iter(per_modality_features))]))
    assert len(set(lengths)) == 1  # make sure each modality has the same sample num
    sample_num = lengths[0]

    return modality_features, modality_types, sample_num


def apply_data_processor(
    per_sample_features: Dict, data_processors: Dict, feature_modalities: Dict, is_training: bool
):
    """
    Process one sample's features.

    Parameters
    ----------
    per_sample_features
        Modality features of one sample.
    data_processors
        A dict of data processors.
    is_training
        Whether is training.

    Returns
    -------
    The processed features of one sample.
    """
    sample_features = {}
    for per_modality, per_modality_processors in data_processors.items():
        for per_model_processor in per_modality_processors:
            if per_modality in per_sample_features and per_sample_features[per_modality]:
                sample_features.update(
                    per_model_processor(
                        per_sample_features[per_modality], feature_modalities[per_modality], is_training=is_training
                    )
                )

    return sample_features


def get_per_sample_features(
    modality_features: Dict, modality_types: Dict, idx: int, id_mappings: Optional[Dict] = None
):
    """
    Extract the modality features of one sample.

    Parameters
    ----------
    modality_features
        Modality features of all samples.
    modality_types
        Data types of all columns.
    idx
        The sample index.
    id_mappings
        Id-to-content mappings. The contents can be text, image, etc.
        This is used when the dataframe contains the query/response indexes instead of their contents.

    Returns
    -------
    One sample's modality features.
    """
    ret = dict()
    for per_modality, per_modality_features in modality_features.items():
        if per_modality_features:
            per_modality_ret = dict()
            for per_col_name, per_col_features in per_modality_features.items():
                per_sample_features = per_col_features[idx]
                if (
                    modality_types
                    and modality_types[per_modality]
                    and modality_types[per_modality][per_col_name].endswith(IDENTIFIER)
                ):
                    per_sample_features = id_mappings[per_col_name][per_sample_features]

                per_modality_ret[per_col_name] = per_sample_features
            ret[per_modality] = per_modality_ret

    return ret


def register_encoding_decoding_error_handlers() -> None:
    """Register the encoding and decoding error handlers for `utf-8` and `cp1252`."""

    def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
        return error.object[error.start : error.end].encode("utf-8"), error.end

    def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
        return error.object[error.start : error.end].decode("cp1252"), error.end

    codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
    codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


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


def process_ner_annotations(ner_annotations, ner_text, tokenizer, is_eval=False):
    """
    Generate token-level/word-level labels with given text and NER annotations.

    Parameters
    ----------
    ner_annotations
        The NER annotations.
    ner_text
        The corresponding raw text.
    tokenizer
        The tokenizer to be used.
    is_eval
        Whether it is for evaluation or not, default: False

    Returns
    -------
    Token-level/word-level labels and text features.
    """
    col_tokens, token_to_word_mappings, word_offsets = tokenize_ner_text(ner_text, tokenizer)
    num_words = len(set(token_to_word_mappings)) - 1
    word_label = [1] * num_words
    # TODO: Potentially optimize word label generation via binary search
    for idx, word_offset in enumerate(word_offsets[:num_words, :]):
        for annot in ner_annotations:
            custom_offset = annot[0]
            custom_label = annot[1]
            # support multiple words in an annotated offset range.
            if word_offset[0] >= custom_offset[0] and word_offset[1] <= custom_offset[1]:
                word_label[idx] = custom_label

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


def tokenize_ner_text(text, tokenizer):
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
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    words_with_offsets = is_space_counted(words_with_offsets)
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
    # token to word mappings: it will tell us which token belongs to which word.
    token_to_word_mappings = [i if i != None else -1 for i in col_tokens.word_ids()]
    if len(set(token_to_word_mappings)) != len(words) + 1:
        warnings.warn(f"The token to word mappings are incorrect!")
    offset_mapping = np.array(col_tokens.offset_mapping, dtype=np.int32)
    word_offsets = np.pad(word_offsets, ((0, offset_mapping.shape[0] - len(words)), (0, 0)), "constant")
    return col_tokens, token_to_word_mappings, word_offsets


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


def is_rois_input(sample):
    """
    check if a sample is rois for object detection

    Parameters
    ----------
    sample
        The sampled data.

    Returns
    -------
    bool, whether a sample is rois for object detection
    """
    return isinstance(sample, list) and len(sample) and isinstance(sample[0], list) and len(sample[0]) == 5
