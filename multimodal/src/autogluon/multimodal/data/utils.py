import ast
import codecs
import copy
import re
import warnings
from io import BytesIO
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import PIL
from omegaconf import ListConfig
from text_unidecode import unidecode
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from tokenizers import pre_tokenizers
from torchvision import transforms

from ..constants import (
    CLIP_IMAGE_MEAN,
    CLIP_IMAGE_STD,
    IDENTIFIER,
    IMAGE,
    IMAGE_BYTEARRAY,
    IMAGE_PATH,
    MMDET_IMAGE,
    MMLAB_MODELS,
)
from .collator import DictCollator
from .preprocess_dataframe import MultiModalFeaturePreprocessor

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
    NEAREST = InterpolationMode.NEAREST
except ImportError:
    BICUBIC = PIL.Image.BICUBIC
    NEAREST = PIL.Image.NEAREST

from .randaug import RandAugment
from .trivial_augmenter import TrivialAugment


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
    per_sample_features: Dict,
    data_processors: Dict,
    feature_modalities: Dict,
    is_training: bool,
    load_only=False,
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
    load_only
        Whether to only load the data. Other processing steps may happen in dataset.__getitem__.

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
                        per_sample_features[per_modality],
                        feature_modalities[per_modality],
                        is_training=is_training,
                        load_only=load_only,
                    )
                    if per_model_processor.prefix.lower().startswith(MMDET_IMAGE)
                    else per_model_processor(
                        per_sample_features[per_modality],
                        feature_modalities[per_modality],
                        is_training=is_training,
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


def process_ner_annotations(ner_annotations, ner_text, entity_map, tokenizer, is_eval=False):
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
    col_tokens, token_to_word_mappings, word_offsets = tokenize_ner_text(ner_text, tokenizer)
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
                    re.match(b_prefix, custom_label, re.IGNORECASE) or re.match(i_prefix, custom_label, re.IGNORECASE)
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
    words_with_offsets = pre_tokenizers.BertPreTokenizer().pre_tokenize_str(text)
    words_with_offsets = is_space_counted(words_with_offsets) if len(words_with_offsets) > 1 else words_with_offsets
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


def get_text_token_max_len(provided_max_len, config, tokenizer, checkpoint_name):
    """
    Compute the allowable max length of token sequences.

    Parameters
    ----------
    provided_max_len
        The provided max length.
    config
        Model config.
    tokenizer
        Text tokenizer.
    checkpoint_name
        Name of checkpoint.

    Returns
    -------
    Token sequence max length.
    """
    if hasattr(config, "relative_attention") and config.relative_attention:
        default_max_len = tokenizer.model_max_length
    elif hasattr(config, "position_embedding_type") and "relative" in config.position_embedding_type:
        default_max_len = tokenizer.model_max_length
    elif hasattr(config, "max_position_embeddings"):
        default_max_len = config.max_position_embeddings
    else:
        default_max_len = tokenizer.model_max_length

    if provided_max_len is None or provided_max_len <= 0:
        max_len = default_max_len
    else:
        if provided_max_len < default_max_len:
            if default_max_len < 10**6:  # Larger than this value usually means infinite.
                warnings.warn(
                    f"provided max length: {provided_max_len} "
                    f"is smaller than {checkpoint_name}'s default: {default_max_len}"
                )
        max_len = min(provided_max_len, default_max_len)

    return max_len


def get_image_transform_funcs(transform_types: Union[List[str], ListConfig, List[Callable]], size: int):
    """
    Parse a list of transform strings into callable objects.

    Parameters
    ----------
    transform_types
        A list of transforms, which can be strings or callable objects.
    size
        Image size.

    Returns
    -------
    A list of transform objects.
    """
    image_transforms = []

    if not transform_types:
        return image_transforms

    if isinstance(transform_types, ListConfig):
        transform_types = list(transform_types)
    elif not isinstance(transform_types, list):
        transform_types = [transform_types]

    if all([isinstance(trans_type, str) for trans_type in transform_types]):
        pass
    elif all([isinstance(trans_type, Callable) for trans_type in transform_types]):
        return copy.copy(transform_types)
    else:
        raise ValueError(f"transform_types {transform_types} contain neither all strings nor all callable objects.")

    for trans_type in transform_types:
        args = None
        kargs = None
        if "(" in trans_type:
            trans_mode = trans_type[0 : trans_type.find("(")]
            if "{" in trans_type:
                kargs = ast.literal_eval(trans_type[trans_type.find("{") : trans_type.rfind(")")])
            else:
                args = ast.literal_eval(trans_type[trans_type.find("(") :])
        else:
            trans_mode = trans_type

        if trans_mode == "resize_to_square":
            image_transforms.append(transforms.Resize((size, size), interpolation=BICUBIC))
        elif trans_mode == "resize_gt_to_square":
            image_transforms.append(transforms.Resize((size, size), interpolation=NEAREST))
        elif trans_mode == "resize_shorter_side":
            image_transforms.append(transforms.Resize(size, interpolation=BICUBIC))
        elif trans_mode == "center_crop":
            image_transforms.append(transforms.CenterCrop(size))
        elif trans_mode == "random_resize_crop":
            image_transforms.append(transforms.RandomResizedCrop(size))
        elif trans_mode == "random_horizontal_flip":
            image_transforms.append(transforms.RandomHorizontalFlip())
        elif trans_mode == "random_vertical_flip":
            image_transforms.append(transforms.RandomVerticalFlip())
        elif trans_mode == "color_jitter":
            if kargs is not None:
                image_transforms.append(transforms.ColorJitter(**kargs))
            elif args is not None:
                image_transforms.append(transforms.ColorJitter(*args))
            else:
                image_transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
        elif trans_mode == "affine":
            if kargs is not None:
                image_transforms.append(transforms.RandomAffine(**kargs))
            elif args is not None:
                image_transforms.append(transforms.RandomAffine(*args))
            else:
                image_transforms.append(transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)))
        elif trans_mode == "randaug":
            if kargs is not None:
                image_transforms.append(RandAugment(**kargs))
            elif args is not None:
                image_transforms.append(RandAugment(*args))
            else:
                image_transforms.append(RandAugment(2, 9))
        elif trans_mode == "trivial_augment":
            image_transforms.append(TrivialAugment(IMAGE, 30))
        else:
            raise ValueError(f"unknown transform type: {trans_mode}")

    return image_transforms


def construct_image_processor(
    image_transforms: Union[List[Callable], List[str]],
    size: int,
    normalization,
) -> transforms.Compose:
    """
    Build up an image processor from the provided list of transform types.

    Parameters
    ----------
    image_transforms
        A list of image transform types.
    size
        Image size.
    normalization
        A transforms.Normalize object. When the image is ground truth image, 'normalization=None' should be specified.

    Returns
    -------
    A transforms.Compose object.
    """
    image_transforms = get_image_transform_funcs(transform_types=image_transforms, size=size)
    if not any([isinstance(trans, transforms.ToTensor) for trans in image_transforms]):
        image_transforms.append(transforms.ToTensor())
    if not any([isinstance(trans, transforms.Normalize) for trans in image_transforms]) and normalization != None:
        image_transforms.append(normalization)
    return transforms.Compose(image_transforms)


def image_mean_std(norm_type: str):
    """
    Get image normalization mean and std by its name.

    Parameters
    ----------
    norm_type
        Name of image normalization.

    Returns
    -------
    Normalization mean and std.
    """
    if norm_type == "inception":
        return IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    elif norm_type == "imagenet":
        return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    elif norm_type == "clip":
        return CLIP_IMAGE_MEAN, CLIP_IMAGE_STD
    else:
        raise ValueError(f"unknown image normalization: {norm_type}")
