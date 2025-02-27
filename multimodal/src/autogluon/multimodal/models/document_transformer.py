import inspect
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

from transformers import logging as hf_logging

from ..constants import (
    ATTENTION_MASK,
    BBOX,
    COLUMN_FEATURES,
    FEATURES,
    IMAGE,
    INPUT_IDS,
    LOGITS,
    MASKS,
    PIXEL_VALUES,
    TOKEN_TYPE_IDS,
)
from .hf_text import HFAutoModelForTextPrediction
from .utils import get_column_features, get_image_size_mean_std

hf_logging.set_verbosity_error()

logger = logging.getLogger(__name__)


class DocumentTransformer(HFAutoModelForTextPrediction):
    """
    Document Classification with Huggingface backbones. Inherit from HFAutoModelForTextPrediction.
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str = "microsoft/layoutlmv3-base",
        num_classes: Optional[int] = 0,
        pooling_mode: Optional[str] = "cls",
        gradient_checkpointing: Optional[bool] = False,
        low_cpu_mem_usage: Optional[bool] = False,
        pretrained: Optional[bool] = True,
        tokenizer_name: Optional[str] = "hf_auto",
        image_size: Optional[int] = None,
        image_norm: Optional[str] = None,
    ):
        """
        Load a pretrained huggingface layout-aware document transformer backbone.

        Parameters
        ----------
        prefix
            The model prefix.
        checkpoint_name
            Name of the checkpoint. We support loading checkpoint from
            Huggingface Models list: https://huggingface.co/models
            For example, you can use layout-aware models:
                - microsoft/layoutlmv3-base
                - microsoft/layoutlm-base-uncased
                - microsoft/xdoc-base
                - microsoft/layoutxlm-base
                - microsoft/layoutlmv2-base-uncased
            you may also use text focused transformers:
                - 'microsoft/deberta-v3-base'
                - 'bert-base-uncased'
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
        tokenizer_name
            Name of the huggingface tokenizer type.
        image_norm
            How to normalize an image. We now support:
            - inception
                Normalize image by IMAGENET_INCEPTION_MEAN and IMAGENET_INCEPTION_STD from timm
            - imagenet
                Normalize image by IMAGENET_DEFAULT_MEAN and IMAGENET_DEFAULT_STD from timm
            - clip
                Normalize image by mean (0.48145466, 0.4578275, 0.40821073) and
                std (0.26862954, 0.26130258, 0.27577711), used for CLIP.
        image_size
            The provided width / height of a square image.
        """
        logger.debug(f"initializing {prefix} (DocumentTransformer)")
        logger.debug(f"model checkpoint: {checkpoint_name}")
        super().__init__(
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            num_classes=num_classes,
            pooling_mode=pooling_mode,
            gradient_checkpointing=gradient_checkpointing,
            low_cpu_mem_usage=low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=tokenizer_name,
        )
        self.image_size, self.image_mean, self.image_std = get_image_size_mean_std(
            model_name=self.prefix,
            config=self.config,
            provided_size=image_size,
            provided_norm_type=image_norm,
        )
        self.is_text_only_flag = self.is_text_only()

        if self.is_text_only_flag:
            logger.debug(f"Checkpoint: {checkpoint_name} uses the text data only for classification.")

    def is_text_only(self):
        """
        Check the tokenizer to see if it is a text only tokenizer.

        Parameters
        ----------
        tokenizer
            The tokenizer to be used.

        Returns
        -------
        True if the tokenizer only accept text, otherwise, False.
        """
        model_args = list(inspect.signature(self.tokenizer.__call__).parameters.keys())
        # Tokenizers of document foundation models usually have a "boxes" argument.
        if "boxes" not in model_args:
            return True
        else:
            return False

    @property
    def text_attention_mask_key(self):
        return f"{self.prefix}_{ATTENTION_MASK}"

    @property
    def text_bbox_key(self):
        return f"{self.prefix}_{BBOX}"

    @property
    def document_pixel_value_key(self):
        return f"{self.prefix}_{PIXEL_VALUES}"

    def update_input_data(
        self,
        input_data: dict,
        batch: dict,
        keys: dict,
    ):
        """
        Update the model input data based on the model argument.
        For example, microsoft/layoutlm-base-uncased has a "bbox" argument;
        microsoft/layoutlmv3-base has arguments: "bbox" and "image".
        Text only bert does not have these two arguments.

        Parameters
        ----------
        input_data
            A dictionary containing the model input data.
        batch:
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.
        keys:
            A dictionary containing the model arguments and corresponding batch keys.
        """
        model_args = list(inspect.signature(self.model.forward).parameters.keys())
        for key, value in keys.items():
            if key in model_args:
                input_data.update({key: batch[value]})

    def forward(
        self,
        batch: dict,
    ):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with logits and features.
        """
        input_data = {}
        self.update_input_data(
            input_data,
            batch,
            keys={
                INPUT_IDS: self.text_token_ids_key,
                TOKEN_TYPE_IDS: self.text_segment_ids_key,
                ATTENTION_MASK: self.text_attention_mask_key,
                BBOX: self.text_bbox_key,
                PIXEL_VALUES: self.document_pixel_value_key,
                IMAGE: self.document_pixel_value_key,
            },
        )

        text_masks = batch[self.text_attention_mask_key]

        outputs = self.model(**input_data)

        if self.pooling_mode == "cls":
            pooled_features = outputs.last_hidden_state[:, 0, :]
        elif self.pooling_mode == "mean":
            # In some models, last_hidden_state is the concatenation of document image features and text features.
            pooled_features = outputs.last_hidden_state.mean(1)
        else:
            raise NotImplementedError(f"Pooling mode={self.pooling_mode} is not supported.")

        logits = self.head(pooled_features)

        ret = {COLUMN_FEATURES: {FEATURES: {}, MASKS: {}}}
        column_features, column_feature_masks = get_column_features(
            batch=batch,
            column_name_prefix=self.text_column_prefix,
            features=outputs.last_hidden_state,
            valid_lengths=sum(text_masks),
            cls_feature=pooled_features,
        )
        ret[COLUMN_FEATURES][FEATURES].update(column_features)
        ret[COLUMN_FEATURES][MASKS].update(column_feature_masks)

        ret.update(
            {
                LOGITS: logits,
                FEATURES: pooled_features,
            }
        )

        return {self.prefix: ret}
