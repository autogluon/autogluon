import torch
import logging
from torch import nn
import warnings
from transformers import AutoModel, AutoTokenizer
from transformers import logging as hf_logging
from ..constants import (
    TEXT_TOKEN_IDS, TEXT_VALID_LENGTH, TEXT_SEGMENT_IDS,
    LABEL, LOGITS, FEATURES, AUTOMM, COLUMN,
)
from typing import Optional, List, Tuple
from .utils import (
    assign_layer_ids,
    init_weights,
    get_column_features,
)

hf_logging.set_verbosity_error()

logger = logging.getLogger(AUTOMM)


class HFAutoModelForTextPrediction(nn.Module):
    """
    Support huggingface text backbones.
    Refer to https://github.com/huggingface/transformers
    """

    def __init__(
            self,
            prefix: str,
            checkpoint_name: str = 'microsoft/deberta-v3-base',
            num_classes: Optional[int] = 0,
    ):
        """
        Load a pretrained huggingface text transformer backbone.

        Parameters
        ----------
        prefix
            The model prefix.
        checkpoint_name
            Name of the checkpoint. We support loading checkpoint from
            Huggingface Models list: https://huggingface.co/models
            For example, you may use
                English backbones:
                    - 'microsoft/deberta-v3-base'
                    - 'bert-base-uncased'
                    - 'google/electra-base-discriminator'
                    - 'distilroberta-base'
                Multilingual backbones:
                    - 'microsoft/mdeberta-v3-base'
                    - 'xlm-roberta-base'
        num_classes
            The number of classes. 1 for a regression task.
        """
        super().__init__()
        logger.debug(f"initializing {checkpoint_name}")
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes
        self.model = AutoModel.from_pretrained(checkpoint_name)
        self.out_features = self.model.config.hidden_size

        self.head = nn.Linear(self.out_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head.apply(init_weights)

        self.prefix = prefix

        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

        if hasattr(self.model.config, 'type_vocab_size') and self.model.config.type_vocab_size <= 1:
            # Disable segment ids for models like RoBERTa
            self.disable_seg_ids = True
        else:
            self.disable_seg_ids = False

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
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    @property
    def text_column_prefix(self):
        return f"{self.text_token_ids_key}_{COLUMN}"

    @property
    def text_feature_dim(self):
        return self.model.config.hidden_size

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
        text_token_ids = batch[self.text_token_ids_key]
        if self.disable_seg_ids:
            text_segment_ids = None
        else:
            text_segment_ids = batch[self.text_segment_ids_key]
        text_valid_length = batch[self.text_valid_length_key]

        steps = torch.arange(0, text_token_ids.shape[1]).type_as(text_valid_length)
        text_masks = (steps.reshape((1, -1)) < text_valid_length.reshape((-1, 1))).type_as(text_token_ids)

        outputs = self.model(
            input_ids=text_token_ids,
            token_type_ids=text_segment_ids,
            attention_mask=text_masks,
        )
        cls_features = outputs.last_hidden_state[:, 0, :]

        logits = self.head(cls_features)

        ret = get_column_features(
            batch=batch,
            column_name_prefix=self.text_column_prefix,
            features=outputs.last_hidden_state,
            valid_lengths=text_valid_length,
        )

        ret.update(
            {
                LOGITS: logits,
                FEATURES: cls_features,
            }
        )

        return {self.prefix: ret}

    def get_layer_ids(self):

        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        In the AutoModel scenario, this function may not always return the correct result.
        Thus, you can use "print(json.dumps(name_to_id, indent=2))" to manually check whether
        the layer ids are reasonable.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = "model"
        pre_encoder_patterns = ("embeddings", "LayerNorm", "wte", "wpe")
        post_encoder_patterns = ("head", "pooler", "ln_f")
        names = [n for n, _ in self.named_parameters()]

        name_to_id, names = assign_layer_ids(
            names=names,
            pre_encoder_patterns=pre_encoder_patterns,
            post_encoder_patterns=post_encoder_patterns,
            model_pre=model_prefix,
        )
        if len(names) > 0:
            logger.debug(f"outer layers are treated as head: {names}")
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 0

        return name_to_id
