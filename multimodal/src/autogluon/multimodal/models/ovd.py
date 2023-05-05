import logging
import os
from typing import Optional

import torch
from torch import nn

from ..constants import (
    AUTOMM,
    BBOX,
    COLUMN,
    COLUMN_FEATURES,
    FEATURES,
    IMAGE,
    IMAGE_META,
    IMAGE_VALID_NUM,
    LABEL,
    LOGIT_SCALE,
    LOGITS,
    MASKS,
    PROMPT,
    TEXT_TOKEN_IDS,
    TEXT_VALID_LENGTH,
)
from .utils import assign_layer_ids, get_column_features, init_weights

logger = logging.getLogger(__name__)


class OVDModel(nn.Module):
    """
    Support the OVD models.
    Now support GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str,
        pretrained: Optional[bool] = True,
    ):
        """
        Load the pretrained CLIP from huggingface transformers.

        Parameters
        ----------
        prefix
            The model prefix.
        num_classes
            The number of classes. 1 for a regression task.
        pretrained
            Whether using the pretrained weights. If pretrained=True, download the pretrained model.
        """
        super().__init__()
        logger.debug(f"initializing {checkpoint_name}")
        self.checkpoint_name = checkpoint_name

        self.config, self.model = self.get_ovd_config_and_model(checkpoint_name=checkpoint_name, pretrained=pretrained)

        self.box_threshold = self.config.box_threshold
        self.text_threshold = self.config.text_threshold

        self.prefix = prefix

        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    @property
    def image_key(self):
        return f"{self.prefix}_{IMAGE}"

    @property
    def image_meta_key(self):
        return f"{self.prefix}_{IMAGE_META}"

    @property
    def prompt_key(self):
        return f"{self.prefix}_{PROMPT}"

    @property
    def image_valid_num_key(self):
        return f"{self.prefix}_{IMAGE_VALID_NUM}"

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    @property
    def text_column_prefix(self):
        return f"{self.text_token_ids_key}_{COLUMN}"

    @property
    def image_column_prefix(self):
        return f"{self.image_key}_{COLUMN}"

    def forward(
        self,
        batch: dict,
        with_logits: bool = False,
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
        from groundingdino.util.utils import get_phrases_from_posmap

        def process_caption(caption):  # TODO: split if it's too long
            caption = caption.lower().strip()
            if not caption.endswith("."):
                caption = caption + "."
            return caption

        captions = [process_caption(caption) for caption in batch[self.prompt_key]]
        images = batch[self.image_key]

        with torch.no_grad():
            outputs = self.model(images, captions=captions)
        logits = (
            outputs["pred_logits"].cpu().sigmoid()
        )  # (bs, nq, 256)  # TODO: will .cpu() affect the data collection of lightning once we support multi gpu?
        boxes = outputs["pred_boxes"].cpu()  # (bs, nq, 4)

        # filter output
        batch_size = logits.shape[0]  # bs
        logits_filt = logits.clone()  # bs, num_filt(900), 256
        boxes_filt = boxes.clone()  # bs, num_filt(900), 4
        filt_mask = logits_filt.max(dim=2)[0] > self.box_threshold  # bs, num_filt(900)
        logits_filt = [
            logits_filt[sample_idx][filt_mask[sample_idx]] for sample_idx in range(batch_size)
        ]  # [bs * (num_after_mask_this_sample, 256)]
        boxes_filt = [
            boxes_filt[sample_idx][filt_mask[sample_idx]] for sample_idx in range(batch_size)
        ]  # [bs * (num_after_mask_this_sample, 4)]

        # get phrase
        tokenizer = self.model.tokenizer
        tokenized = [tokenizer(caption) for caption in captions]
        # build pred
        pred_phrases = []
        for sample_idx in range(batch_size):
            pred_phrase_per_sample = []
            logits_filt_per_sample = logits_filt[sample_idx]
            boxes_filt_per_sample = boxes_filt[sample_idx]
            tokenized_per_sample = tokenized[sample_idx]
            for logit, box in zip(logits_filt_per_sample, boxes_filt_per_sample):
                pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized_per_sample, tokenizer)
                if with_logits:
                    pred_phrase_per_sample.append(pred_phrase + f"({str(logit.max().item())[:4]})")
                else:
                    pred_phrase_per_sample.append(pred_phrase)
            pred_phrases.append(pred_phrase_per_sample)

        ret = {
            BBOX: boxes_filt,
            PROMPT: pred_phrases,
            LOGITS: [logits_filt_per_sample.max(dim=1)[0] for logits_filt_per_sample in logits_filt],
            IMAGE_META: batch[self.image_meta_key],
        }

        return {self.prefix: ret}

    def get_ovd_config_and_model(
        self,
        checkpoint_name: str,
        pretrained: bool = True,
    ):
        from ..utils import download, get_pretrain_configs_dir

        ovd_configs_dir = get_pretrain_configs_dir(subfolder="ovd")
        OVD_CUSTOM_MODELS = {
            "GroundingDINO_SwinB": {
                "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",
                "config_file": os.path.join(ovd_configs_dir, "grounding_dino", "GroundingDINO_SwinB.cfg.py"),
            },
            "GroundingDINO_SwinT_OGC": {
                "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
                "config_file": os.path.join(ovd_configs_dir, "grounding_dino", "GroundingDINO_SwinT_OGC.py"),
            },
        }

        if checkpoint_name in ["GroundingDINO_SwinB", "GroundingDINO_SwinT_OGC"]:
            try:
                from groundingdino.models import build_model
                from groundingdino.util.slconfig import SLConfig
                from groundingdino.util.utils import clean_state_dict
            except ImportError as e:
                raise ImportError(
                    "Please install groundingdino by: pip install git+https://github.com/IDEA-Research/GroundingDINO.git"
                )

            model_config_path = OVD_CUSTOM_MODELS[checkpoint_name]["config_file"]

            if checkpoint_name in OVD_CUSTOM_MODELS:
                # TODO: add sha1_hash
                model_checkpoint_path = download(
                    url=OVD_CUSTOM_MODELS[checkpoint_name]["url"],
                )
            else:
                raise ValueError(
                    f"Checkpoint name not available {checkpoint_name}. All supported OVD models are {OVD_CUSTOM_MODELS.keys()}"
                )

            config = SLConfig.fromfile(model_config_path)
            config.device = "cpu"  # lightning will handle the device part
            model = build_model(config)
            if pretrained:
                checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
                load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
                logger.debug(f"load_res: \n{load_res}")
        else:
            raise ValueError(f"checkpoint name {checkpoint_name} not supported.")

        return config, model

    def get_layer_ids(
        self,
    ):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        # TODO

        return {}
