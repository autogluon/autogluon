import codecs
import random
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd
from nlpaug import Augmenter
from nlpaug.util import Method
from text_unidecode import unidecode

from ..constants import MMCV_MODELS
from .collator import Dict
from .preprocess_dataframe import MultiModalFeaturePreprocessor


def extract_value_from_config(
    config: dict,
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
    data_processors: Union[dict, List[dict]],
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
                    if per_model_processor.prefix.lower().startswith(MMCV_MODELS):
                        collate_fn.update(
                            per_model_processor.collate_fn(
                                per_modality_column_names, per_gpu_batch_size=per_gpu_batch_size
                            )
                        )
                    else:
                        collate_fn.update(per_model_processor.collate_fn(per_modality_column_names))
    return Dict(collate_fn)


def apply_df_preprocessor(data: pd.DataFrame, df_preprocessor: MultiModalFeaturePreprocessor, modalities: Iterable):
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
    sample_num
        Number of samples.
    """
    lengths = []
    modality_features = {}
    for per_modality in modalities:
        per_modality_features = getattr(df_preprocessor, f"transform_{per_modality}")(data)
        modality_features[per_modality] = per_modality_features
        if per_modality_features:
            lengths.append(len(per_modality_features[next(iter(per_modality_features))]))
    assert len(set(lengths)) == 1  # make sure each modality has the same sample num
    sample_num = lengths[0]

    return modality_features, sample_num


def apply_data_processor(modality_features: dict, data_processors: dict, idx: int, is_training: bool):
    """
    Process one sample's features.

    Parameters
    ----------
    modality_features
        Features of different modalities got from `apply_df_preprocessor`.
    data_processors
        A dict of data processors.
    idx
        The sample index.
    is_training
        Whether is training.

    Returns
    -------
    The processed features of one sample.
    """
    sample_features = {}
    for per_modality, per_modality_processors in data_processors.items():
        for per_model_processor in per_modality_processors:
            if modality_features[per_modality]:
                sample_features.update(
                    per_model_processor(modality_features[per_modality], idx=idx, is_training=is_training)
                )

    return sample_features


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
